import argparse
from data_utils import custom_load_dataset
from utils import *
import math
import gym
from gym import spaces
import copy

import string
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize
from supar import Parser
from nltk.tokenize.treebank import TreebankWordDetokenizer
from utils import setup_roberta, setup_gpt2, setup_t5

# Editing the instruction
def delete_phrase(candidate, phrase):
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ')
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', ' ')
    else: 
        answer = candidate.replace(phrase, '')
    return answer

def add_phrase(candidate, phrase, after):
    if after == '': answer = phrase + ' ' + candidate
    else: 
        if candidate.find(' ' + after) > 0:
            answer = candidate.replace(' ' + after, ' ' + after + ' ' + phrase)
        elif candidate.find(after + ' ') > 0:
            answer = candidate.replace(after + ' ', after + ' ' + phrase + ' ')
        else: 
            answer = candidate.replace(after, after + phrase )
    return answer

def swap_phrases(candidate, phrase_1, phrase_2):
    if candidate.find(' ' + phrase_1 + ' ') >= 0 : 
        candidate = candidate.replace(' ' + phrase_1 + ' ', ' <1> ')
    else: candidate = candidate.replace(phrase_1, '<1>')
    if candidate.find(' ' + phrase_2 + ' ') >= 0 : 
        candidate = candidate.replace(' ' + phrase_2 + ' ', ' <2> ')
    else: candidate = candidate.replace(phrase_2, '<2>')
    candidate = candidate.replace('<1>', phrase_2)
    candidate = candidate.replace('<2>', phrase_1)
    return candidate

def substitute_phrase(candidate, phrase):
    num_beams = 10
    num_return_sequences = 10
    paraphrases = get_response(phrase, num_return_sequences, num_beams)
    paraphrase = np.random.choice(paraphrases, 1)[0] 
    paraphrase = paraphrase.strip('.')
    if candidate.find(' ' + phrase) > 0:
        answer = candidate.replace(' ' + phrase, ' ' + paraphrase)
    elif candidate.find(phrase + ' ') > 0:
        answer = candidate.replace(phrase + ' ', paraphrase + ' ')
    else: 
        answer = candidate.replace(phrase, paraphrase)
    return answer

def perform_edit(edit, base, phrase_lookup, delete_tracker):
    if edit == 'del':
        [i] = np.random.choice(list(phrase_lookup.keys()), 1) 
        return delete_phrase(base, phrase_lookup[i]), [i]
    elif edit == 'swap':
        try: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=False) 
        except: [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=True) 
        return swap_phrases(base, phrase_lookup[i], phrase_lookup[j]), [i, j]
    elif edit == 'sub':
        [i] = np.random.choice(list(phrase_lookup.keys()), 1) 
        return substitute_phrase(base, phrase_lookup[i]), [i]
    elif edit == 'add':
        keys = list(phrase_lookup.keys())
        keys.append(-1)
        [i] = np.random.choice(keys, 1) 
        if i >= 0: after = phrase_lookup[i]
        else: after = ''
        if len(delete_tracker) == 0: return base, []
        phrase = np.random.choice(delete_tracker, 1)[0]
        return add_phrase(base, phrase, after), [phrase]

# Tokenize the sentence
def traverse_tree(parsed_tree):
    phrases = []
    for tree in parsed_tree:
        if tree.label() == '_': continue
        phrases.append(detokenize(tree.leaves()))
        for subtree in tree:
            if type(subtree) == nltk.tree.Tree:
                if subtree.label() == '_': continue
                phrases.append(detokenize(subtree.leaves()))
                phrases.extend(traverse_tree(subtree))
    return phrases

def check_child(tree):
    check = False
    count = 0
    total_count = 0
    for subtree in tree:
        total_count += 1
        if type(subtree) == nltk.tree.Tree:
            if subtree.label() == '_':
                count += 1
    if count >= total_count - count: check = True

    return check

def collect_leaves(parsed_tree):
    leaves = []
    for tree in parsed_tree:
        if type(parsed_tree) != nltk.tree.Tree: continue
        if tree.label() == '_': 
            leaves.append(detokenize(tree.leaves()))
            continue
        if check_child(tree): leaves.append(detokenize(tree.leaves()))
        else:
            leaves.extend(collect_leaves(tree))
    return leaves

def get_phrases(instruction, parser): # one possible way of obtaining disjoint phrases
    phrases = []
    for sentence in sent_tokenize(instruction):
        parsed_tree = parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
        leaves = collect_leaves(parsed_tree)
        phrases.extend(leaves)
    phrases = [detokenize(word_tokenize(phrase)) for phrase in phrases if phrase not in string.punctuation or phrase == '']
    return phrases

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    total_list = []
    tp = []
    fp = []
    fn = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
        # print(ans_label, true_label, flush=True)
        if ans_label == true_label and true_label == 1:
            tp.append(1)
        if ans_label != true_label and ans_label == 1:
            fp.append(1)
        if ans_label != true_label and ans_label == 0:
            fn.append(1)
        total_list.append(1)
    return np.sum(correctness_list), np.sum(total_list), np.sum(tp), np.sum(fp), np.sum(fn)

def get_label_probs(params, raw_resp, train_sentences, train_labels, test_sentences):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(params['label_dict'])
    approx = params['approx']
    assert len(raw_resp) == len(test_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    for i, ans in enumerate(raw_resp):
        top_logprobs = ans['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token
        label_probs = [0] * len(params['label_dict'].keys())
        #TODO: changes here
        # label_probs = [1e-12] * len(params['label_dict'].keys())
        for j, label_list in params['label_dict'].items():
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = " " + label  # notice prompt does not have space after 'A:'
                if label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                else:
                    all_found = False
            if not all_found:
                position = (i, j) # (which test example, which label)
                all_missing_positions.append(position)
                # TODO: change this
                label_probs = [1/len(params['label_dict'].keys())] * len(params['label_dict'].keys())
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs) # prob not normalized

    # Fill in the label probs that are NOT in top k probs, by asking the model to rate perplexity
    # This helps a lot in zero shot as most labels wil not be in Top 100 tokens returned by LM
    if (not approx) and (len(all_missing_positions) > 0):
        print(f"Missing probs: {len(all_missing_positions)}/{len(raw_resp) * num_classes}")
        all_additional_prompts = []
        num_prompts_each = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            test_sentence = test_sentences[which_sentence]
            label_list = params['label_dict'][which_label]
            for label in label_list:
                prompt = construct_prompt(params, train_sentences, train_labels, test_sentence)
                prompt += " " + label
                all_additional_prompts.append(prompt)
            num_prompts_each.append(len(label_list))

        # chunk the prompts and feed into model
        chunked_prompts = list(chunks(all_additional_prompts, chunk_size_helper(params)))
        all_probs = []
        for chunk_id, chunk in enumerate(chunked_prompts):
            resp = complete(chunk, 0, params['model'], echo=True, num_log_probs=1)
            for ans in resp['choices']:
                prob = np.exp(ans['logprobs']['token_logprobs'][-1])
                all_probs.append(prob)

        assert sum(num_prompts_each) == len(all_probs)
        assert len(num_prompts_each) == len(all_missing_positions)

        # fill in corresponding entries in all_label_probs
        for index, num in enumerate(num_prompts_each):
            probs = []
            while num > 0:
                probs.append(all_probs.pop(0))
                num -= 1
            prob = np.sum(probs)
            i, j = all_missing_positions[index]
            all_label_probs[i][j] = prob

        assert len(all_probs) == 0, "all should be popped"
        assert (all_label_probs > 0).all(), "all should be populated with non-zero value"

    return all_label_probs # NOT NORMALIZED


def get_phrase_lookup(base_candidate, parser):
    return {p:phrase for p, phrase in enumerate(get_phrases(base_candidate, parser))}
    # Not used for now
    if args.level == 'phrase': phrase_lookup = {p:phrase for p, phrase in enumerate(get_phrases(base_candidate))}
    elif args.level == 'word': 
        words = word_tokenize(base_candidate)
        words = [w for w in words if w not in string.punctuation or w != '']
        phrase_lookup = {p:phrase for p, phrase in enumerate(words)}
    elif args.level == 'sentence':
        sentences = sent_tokenize(base_candidate)
        phrase_lookup = {p:phrase for p, phrase in enumerate(sentences)}
    elif args.level == 'span':
        phrases = []
        for sentence in sent_tokenize(base_candidate):
            spans_per_sentence = np.random.choice(range(2,5)) # split sentence into 2, 3, 4, 5 chunks
            spans = np.array_split(word_tokenize(sentence), spans_per_sentence)
            spans = [detokenize(s) for s in spans]
            phrases.extend(spans)
        phrase_lookup = {p:phrase for p, phrase in enumerate(phrases)}
    else: raise ValueError()
    return phrase_lookup

def detokenize(tokens):
    return TreebankWordDetokenizer().detokenize(tokens)

# This environment supports parallel
class LMForwardEnvNoPrefix(gym.Env):
  """Custom Environment that follows gym interface"""

  def __init__(self, params, prompt_sentence_pool, prompt_label_pool, all_prompt_sentence_pool, all_prompt_label_pool, add_prompt_sentence_pool, add_prompt_label_pool, train_sentences, train_labels, max_steps, num_processes, obs_size, gpu_id=0, entropy_coef=0, loss_type='ce', verbalizer=False, evaluate=False):
    super(LMForwardEnvNoPrefix, self).__init__()
    self.params = params
    self.prompt_sentence_pool = prompt_sentence_pool
    self.prompt_label_pool = prompt_label_pool
    self.all_prompt_sentence_pool = all_prompt_sentence_pool
    self.all_prompt_label_pool = all_prompt_label_pool
    self.add_prompt_sentence_pool = add_prompt_sentence_pool
    self.add_prompt_label_pool = add_prompt_label_pool
    self.train_sentences = train_sentences
    self.train_labels = train_labels
    self.current_prompt = self.prompt_sentence_pool
    self.current_prompt_labels = self.prompt_label_pool
    self.deleted_prompt = []
    self.deleted_prompt_labels = []
    self.latent_type = 'embedding'
    self.loss_type = loss_type
    self.max_steps = max_steps
    self.subset_size = num_processes
    self.num_processes = num_processes
    self.evaluate = evaluate
    self.rew_scale = 100.0
    self.entropy_coef = entropy_coef
    self.verbalizer = verbalizer
    self.correct_bonus = 2.0
    self.incorrect_bonus = 1.8
    self.terminate = []
    if 'gpt2' in params['model']:
        self.model, self.tokenizer = setup_gpt2(params['model'], gpu_id)
    elif 'roberta' in params['model']:
        self.model, self.tokenizer = setup_roberta(params['model'], gpu_id)
    elif 't5' in params['model']:
        self.model, self.tokenizer = setup_t5(params['model'], gpu_id)
    else:
        assert False


    # Prefix editing
    parser = Parser.load('crf-con-en')
    prefix_candidate = detokenize(word_tokenize("The task is to do sentiment analysis"))
    print(get_phrase_lookup(prefix_candidate, parser), flush=True)
    '''
    self.prefix_candidate = detokenize(word_tokenize(params['prompt_prefix']))
    self.parser = Parser.load('crf-con-en')
    self.phrase_lookup = get_phrase_lookup(self.prefix_candidate, self.parser)
    self.phrase_length = len(self.phrase_lookup)
    self.phrase_total_length = int(self.phrase_length*(self.phrase_length-1)/2)
    self.prefix_phrase_total_length = self.phrase_total_length + int(params['num_shots']*(params['num_shots']-1)/2)+1+2
    '''
    self.prompt_swap_length = int(params['num_shots']*(params['num_shots']-1)/2) + 1 + params['num_shots']
    self.prefix_phrase_total_length = self.prompt_swap_length + params['num_shots'] * (params['example_pool_size'] - params['num_shots'])

    self.current_sentence = None
    self.current_label = None
    self.previous_loss = None
    self.idxs = None
    self.steps = 0

    self.swap_idxs1 = []
    self.swap_idxs2 = []
    self.swap_idxs1.append(0)
    self.swap_idxs2.append(0)
    for i in range(params['num_shots']):
        for j in range(i+1, params['num_shots']):
            self.swap_idxs1.append(i)
            self.swap_idxs2.append(j)
    
    for i in range(params['num_shots']):
        self.swap_idxs1.append(i)
        self.swap_idxs2.append(i)
    # 2 for adding and deleting
    '''
    self.swap_idxs1.append(-1)
    self.swap_idxs2.append(-1)
    self.swap_idxs1.append(-2)
    self.swap_idxs2.append(-2)
    '''

    # swap current prompt with pool
    for i in range(params['num_shots']):
        for j in range(params['example_pool_size'] - params['num_shots']):
            self.swap_idxs1.append(i)
            self.swap_idxs2.append(j)
    
    # Prefix indexs
    '''
    self.swap_prefix_idxs1 = []
    self.swap_prefix_idxs2 = []
    self.swap_prefix_idxs1.append(0)
    self.swap_prefix_idxs2.append(0)
    for i in range(self.phrase_length):
        for j in range(i+1, self.phrase_length):
            self.swap_prefix_idxs1.append(i)
            self.swap_prefix_idxs2.append(j)
    '''

    # 3 Verbalizer Dataset formating
    from promptsource.templates import DatasetTemplates
    if params['dataset'] == 'customer_review':
        self.prompt_templates = DatasetTemplates('glue/sst2')
    else:
        self.prompt_templates = DatasetTemplates(params['dataset'])
    self.prompt_template_keys = self.prompt_templates.all_template_names
    print(len(self.prompt_template_keys))
    for key in self.prompt_template_keys:
        answer_lists = self.prompt_templates[key].answer_choices.split("|||")
        for promt_answer, correct_answer in zip(answer_lists, self.params['inv_label_dict'].keys()):
            # print(self.prompt_templates[key].jinja, promt_answer, correct_answer)
            self.prompt_templates[key].jinja = self.prompt_templates[key].jinja.replace(promt_answer.strip(), correct_answer)
            # print(self.prompt_templates[key].jinja)
    self.current_verbalizer = []
    self.deleted_verbalizer = []
    self.subset_verbalizer = []
    self.prefix_phrase_verbalizer_total_length = self.prefix_phrase_total_length + len(self.prompt_template_keys)*params['num_shots']
    # self.prefix_phrase_verbalizer_total_length = self.prefix_phrase_total_length


    # from datasets import load_dataset
    # dataset = load_dataset('super_glue', 'rte', split='train')
    # dataset = load_dataset('trec', split='train')
    '''
    print(dataset[0])
    print(self.prompt_templates[self.prompt_template_keys[0]].apply(dataset[0]))
    exit()
    '''

    print('action space: ', self.prefix_phrase_verbalizer_total_length)
    if self.verbalizer:
        # self.action_space = spaces.Discrete(self.prefix_phrase_verbalizer_total_length + len(self.prompt_template_keys))
        self.action_space = spaces.Discrete(self.prefix_phrase_verbalizer_total_length)
    else:
        self.action_space = spaces.Discrete(self.prefix_phrase_total_length)
    # self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size * 2 + self.max_steps + 1,))
    self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size * (params['example_pool_size'] + 1 - params['num_shots'] + params['num_shots'] * len(self.prompt_template_keys)) + 3,))
    # self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size * (params['example_pool_size'] + 1 + params['example_pool_size'] * len(self.prompt_template_keys)),))
    self.embedding_prepared = torch.tensor(np.array([False])).share_memory_()
    self.current_prompt_embedding_pool = torch.zeros((len(self.train_sentences), params['num_shots'], obs_size)).share_memory_()
    self.add_current_prompt_embedding_pool = torch.zeros((len(self.train_sentences), params['example_pool_size'] - params['num_shots'], obs_size)).share_memory_()
    self.current_verbalizer_embedding_pool = torch.zeros((len(self.train_sentences), params['num_shots'], len(self.prompt_template_keys), obs_size)).share_memory_()
    self.add_current_verbalizer_embedding_pool = torch.zeros((len(self.train_sentences), params['example_pool_size'] - params['num_shots'], len(self.prompt_template_keys), obs_size)).share_memory_()
    if not self.evaluate:
        self.prepare_embedding()

  def prepare_embedding(self):
    print('Preparing Embedding', flush=True)
    prompt_sentence_pool = [copy.deepcopy(self.prompt_sentence_pool) for _ in range(len(self.train_sentences))]
    prompt_label_pool = [copy.deepcopy(self.prompt_label_pool) for _ in range(len(self.train_sentences))]
    add_prompt_sentence_pool = [copy.deepcopy(self.add_prompt_sentence_pool) for _ in range(len(self.train_sentences))]
    add_prompt_label_pool = [copy.deepcopy(self.add_prompt_label_pool) for _ in range(len(self.train_sentences))]
    current_verbalizer_pool = [[0 for _ in range(len(self.prompt_sentence_pool))] for _ in range(len(self.train_sentences))]
    add_current_verbalizer_pool = [[0 for _ in range(self.params['example_pool_size'] - len(self.prompt_sentence_pool))] for _ in range(len(self.train_sentences))]
    subset_verbalizer_pool = [0 for _ in range(len(self.train_sentences))]

    self._current_prompt_embedding_pool = self.embedding(prompt_sentence_pool, prompt_label_pool, current_verbalizer_pool, self.train_sentences, subset_verbalizer_pool)
    self._add_current_prompt_embedding_pool = self.embedding(add_prompt_sentence_pool, add_prompt_label_pool, add_current_verbalizer_pool, self.train_sentences, subset_verbalizer_pool)
    self._current_verbalizer_embedding_pool = []
    for verbalizer in range(len(self.prompt_template_keys)):
        self._current_verbalizer_embedding_pool.append(self.embedding(prompt_sentence_pool, prompt_label_pool, (np.array(current_verbalizer_pool)+verbalizer).tolist(), self.train_sentences, subset_verbalizer_pool))
    self._current_verbalizer_embedding_pool = np.transpose(np.array(self._current_verbalizer_embedding_pool), axes=(1, 2, 0, 3)).tolist()
    self._add_current_verbalizer_embedding_pool = []
    for verbalizer in range(len(self.prompt_template_keys)):
        self._add_current_verbalizer_embedding_pool.append(self.embedding(add_prompt_sentence_pool, add_prompt_label_pool, (np.array(add_current_verbalizer_pool)+verbalizer).tolist(), self.train_sentences, subset_verbalizer_pool))
    self._add_current_verbalizer_embedding_pool = np.transpose(np.array(self._add_current_verbalizer_embedding_pool), axes=(1, 2, 0, 3)).tolist()
    self.current_prompt_embedding_pool[:] = torch.tensor(self._current_prompt_embedding_pool)
    self.add_current_prompt_embedding_pool[:] = torch.tensor(self._add_current_prompt_embedding_pool)
    self.current_verbalizer_embedding_pool[:] = torch.tensor(self._current_verbalizer_embedding_pool)
    self.add_current_verbalizer_embedding_pool[:] = torch.tensor(self._add_current_verbalizer_embedding_pool)

    print(len(self._add_current_prompt_embedding_pool), np.array(self._add_current_prompt_embedding_pool[0]).shape)
    print(len(self._current_prompt_embedding_pool), np.array(self._current_prompt_embedding_pool[0]).shape)
    print(len(self._current_verbalizer_embedding_pool), np.array(self._current_verbalizer_embedding_pool[0]).shape)
    print(len(self._add_current_verbalizer_embedding_pool), np.array(self._add_current_verbalizer_embedding_pool[0]).shape)
    print('Finish Preparing Embedding', flush=True)
    self.embedding_prepared[:] = torch.tensor(np.array([True]))
    # print(len(self.add_current_verbalizer_embedding), len(self.add_current_verbalizer_embedding[0]), len(self.add_current_verbalizer_embedding[0][0]))

  def load_ckpt(self, file_path, i, num_test):
    _current_prompt_embedding_pool = torch.load(file_path+'current_prompt_embedding_pool.pth')
    _add_current_prompt_embedding_pool = torch.load(file_path+'add_current_prompt_embedding_pool.pth')
    _current_verbalizer_embedding_pool = torch.load(file_path+'current_verbalizer_embedding_pool.pth')
    _add_current_verbalizer_embedding_pool = torch.load(file_path+'add_current_verbalizer_embedding_pool.pth')
    print(_current_prompt_embedding_pool.shape, self.current_prompt_embedding_pool.shape)
    self.current_prompt_embedding_pool[:] = _current_prompt_embedding_pool[i*num_test:(i+1)*num_test]
    self.add_current_prompt_embedding_pool[:] = _add_current_prompt_embedding_pool[i*num_test:(i+1)*num_test]
    self.current_verbalizer_embedding_pool[:] = _current_verbalizer_embedding_pool[i*num_test:(i+1)*num_test]
    self.add_current_verbalizer_embedding_pool[:] = _add_current_verbalizer_embedding_pool[i*num_test:(i+1)*num_test]
    print('Finish Preparing Embedding', flush=True)
    self.embedding_prepared[:] = torch.tensor(np.array([True]))

  def get_obs(self, obs, actions):
      # return obs
      # all_obs = np.concatenate([obs, np.array(self.current_prompt_embedding).reshape(obs.shape[0], -1)], axis=-1)
      all_obs = obs
      all_obs = np.concatenate([all_obs, np.array(self.add_current_prompt_embedding).reshape(all_obs.shape[0], -1)], axis=-1)
      # TODO: changes here
      all_obs = np.concatenate([all_obs, np.array(self.current_verbalizer_embedding).reshape(all_obs.shape[0], -1)], axis=-1)
      # all_obs = np.concatenate([all_obs, np.array(self.add_current_verbalizer_embedding).reshape(all_obs.shape[0], -1)], axis=-1)
      all_obs = np.concatenate([all_obs, np.expand_dims(np.array(self.terminate).astype(float)*0+self.steps, -1)], axis=-1)
      all_obs = np.concatenate([all_obs, np.expand_dims(np.array(self.terminate).astype(float), -1)], axis=-1)
      all_obs = np.concatenate([all_obs, np.array(actions).reshape(all_obs.shape[0], -1)], axis=-1)
      return all_obs
      if actions is not None:
          for action, act_history in zip(actions, self.act_histories):
              act_history[self.steps] = action
      new_obs = np.concatenate([obs, self.act_histories], axis=-1)
      return new_obs

  def verbalize(self, current_sentences, current_verbalizer, subset=False):
    if subset: 
        return_sentences = []
        for sentences, verbalizer in zip(current_sentences, current_verbalizer):
            prompt = self.prompt_templates[self.prompt_template_keys[verbalizer]]
            return_sentences.append(prompt.apply(sentences)[0])
        return return_sentences
    else:
        return_sentences = []
        for sentences, verbalizer in zip(current_sentences, current_verbalizer):
            return_sentences.append([])
            for i, sentence in enumerate(sentences):
                prompt = self.prompt_templates[self.prompt_template_keys[verbalizer[i]]]
                # print(sentence)
                # data = {'sentence': sentence, 'text': sentence, 'label': label, 'label-coarse': label, 'label-fine': label}
                # print('new ', prompt.apply(sentence)[0])
                # exit()
                return_sentences[-1].append(prompt.apply(sentence)[0])
        return return_sentences

  def step(self, action):
    # Execute one time step within the environment
    # print(action.shape)
    action = action.squeeze(-1)
    # swap_idx1 = [self.swap_idxs1[act] for act in action]
    # swap_idx2 = [self.swap_idxs2[act] for act in action]
    # print(self.current_prompt)
    idx = 0 
    # import time
    # time_t1 = time.time()
    # print('bef ', [np.mean(embedding) for embedding in self.current_prompt_embedding[0]])
    # print('bef ', self.current_prompt_labels[0])
    # print('bef ', self.current_prompt_index[0], self.current_verbalizer[0])
    for terminate, act, sentence_index, sentence, label, embedding, ver_embedding, add_sentence_index, add_sentence, add_label, add_embedding, add_ver_embedding, delete_sentence, delete_label, delete_embedding, delete_ver_embedding, verbalizer, add_verbalizer, delete_verbalizer, subset_verbalizer in \
        zip(self.terminate, action.tolist(), self.current_prompt_index, self.current_prompt, self.current_prompt_labels, self.current_prompt_embedding, self.current_verbalizer_embedding, self.add_current_prompt_index, self.add_current_prompt, self.add_current_prompt_labels, \
            self.add_current_prompt_embedding, self.add_current_verbalizer_embedding, self.deleted_prompt, self.deleted_prompt_labels, self.deleted_prompt_embedding, self.deleted_verbalizer_embedding, self.current_verbalizer, self.add_current_verbalizer, self.deleted_verbalizer, self.subset_verbalizer):
        # print(idx1, idx2, len(sentence), len(label), len(delete_sentence), len(delete_label))
        if not terminate: 
            if act < self.prefix_phrase_total_length:
                #TODO: maybe we need to swap verbalizer as we swap example
                idx1 = self.swap_idxs1[act]
                idx2 = self.swap_idxs2[act]
                '''
                if idx1 == -1 and idx2 == -1 and len(sentence) > 0 and len(label) > 0:
                    delete_sentence.append(copy.deepcopy(sentence[0]))
                    delete_label.append(copy.deepcopy(label[0]))
                    delete_verbalizer.append(copy.deepcopy(verbalizer[0]))
                    sentence.pop(0)
                    label.pop(0)
                    verbalizer.pop(0)
                elif idx1 == -2 and idx2 == -2 and len(delete_sentence) > 0 and len(delete_label) > 0:
                    sentence.append(copy.deepcopy(delete_sentence[0]))
                    label.append(copy.deepcopy(delete_label[0]))
                    verbalizer.append(copy.deepcopy(delete_verbalizer[0]))
                    delete_sentence.pop(0)
                    delete_label.pop(0)
                    delete_verbalizer.pop(0)
                elif idx1 >= 0 and idx2 >= 0 and act < self.prompt_swap_length:
                '''
                if idx1 == idx2:
                    self.terminate[idx] = True
                if idx1 >= 0 and idx2 >= 0 and act < self.prompt_swap_length:
                    if idx1 < len(sentence) and idx2 < len(sentence):
                        sentence[idx1], sentence[idx2] = copy.deepcopy(sentence[idx2]), copy.deepcopy(sentence[idx1])
                        label[idx1], label[idx2] = copy.deepcopy(label[idx2]), copy.deepcopy(label[idx1])
                        embedding[idx1], embedding[idx2] = copy.deepcopy(embedding[idx2]), copy.deepcopy(embedding[idx1])
                        ver_embedding[idx1], ver_embedding[idx2] = copy.deepcopy(ver_embedding[idx2]), copy.deepcopy(ver_embedding[idx1])
                        verbalizer[idx1], verbalizer[idx2] = copy.deepcopy(verbalizer[idx2]), copy.deepcopy(verbalizer[idx1])
                        sentence_index[idx1], sentence_index[idx2] = copy.deepcopy(sentence_index[idx2]), copy.deepcopy(sentence_index[idx1])
                    else:
                        print('case 1 ', idx1, idx2, len(sentence), len(add_sentence))
                        exit()
                elif idx1 >= 0 and idx2 >= 0:
                    if idx1 < len(sentence) and idx2 < len(add_sentence):
                        sentence[idx1], add_sentence[idx2] = copy.deepcopy(add_sentence[idx2]), copy.deepcopy(sentence[idx1])
                        label[idx1], add_label[idx2] = copy.deepcopy(add_label[idx2]), copy.deepcopy(label[idx1])
                        embedding[idx1], add_embedding[idx2] = copy.deepcopy(add_embedding[idx2]), copy.deepcopy(embedding[idx1])
                        ver_embedding[idx1], add_ver_embedding[idx2] = copy.deepcopy(add_ver_embedding[idx2]), copy.deepcopy(ver_embedding[idx1])
                        # verbalizer[idx1], add_verbalizer[idx2] = copy.deepcopy(add_verbalizer[idx2]), copy.deepcopy(verbalizer[idx1])
                        sentence_index[idx1], add_sentence_index[idx2] = copy.deepcopy(add_sentence_index[idx2]), copy.deepcopy(sentence_index[idx1])
                    else:
                        print('case 2', idx1, idx2, len(sentence), len(add_sentence))
                        exit()
            #TODO: comment out for now
            elif self.verbalizer and act < self.prefix_phrase_verbalizer_total_length:
                act = act - self.prefix_phrase_total_length
                verbalize_idx = act % self.params['num_shots']
                if act == len(self.prompt_template_keys)*self.params['num_shots']:
                    assert False
                elif verbalize_idx < len(verbalizer):
                    verbalizer[verbalize_idx] = int(act / self.params['num_shots'])
                    embedding[verbalize_idx] = copy.deepcopy(np.array(ver_embedding)[verbalize_idx, int(act / self.params['num_shots'])])
                else:
                    assert False
            '''
            elif self.verbalizer:
                act = act - self.prefix_phrase_verbalizer_total_length
                if act < len(self.prompt_template_keys):
                    subset_verbalizer = act
                else:
                    assert False
            '''

        idx += 1
    # print('aft ', [np.mean(embedding) for embedding in self.current_prompt_embedding[0]])
    # print('aft ', self.current_prompt_labels[0])
    # print('aft ', self.current_verbalizer[0])
    # print('aft ', self.current_prompt_index[0], self.current_verbalizer[0])

    if self.verbalizer:
        verbalized_prompt = self.verbalize(self.current_prompt, self.current_verbalizer)
        verbalized_pool = self.verbalize(self.add_current_prompt, self.add_current_verbalizer)
        subset_sentences = self.verbalize(self.subset_sentences, self.subset_verbalizer, subset=True)
    else:
        verbalized_prompt = self.current_prompt
        verbalized_pool = self.add_prompt_sentence_pool
        subset_sentences = self.subset_sentences
    # time1 = time.time()
    raw_resp, obs = get_model_response_parallel(self.params, self.model, self.tokenizer, verbalized_prompt, self.current_prompt_labels, subset_sentences)
    # time2 = time.time()
    # print('model ', time2 - time1)
    # _, add_obs = get_model_response_parallel(self.params, verbalized_pool, self.add_current_prompt_labels, subset_sentences)
    # obs = np.concatenate([obs, add_obs], axis=-1)
    all_label_probs = get_label_probs(self.params, raw_resp, verbalized_prompt, self.current_prompt_labels, subset_sentences)

    assert len(all_label_probs) == len(self.subset_labels)
    # label_probs = all_label_probs / (np.sum(all_label_probs, axis=-1, keepdims=True) + (np.sum(all_label_probs, axis=-1, keepdims=True) == 0))
    label_probs = all_label_probs / np.sum(all_label_probs, axis=-1, keepdims=True)
    # time1 = time.time()
    
    self.steps += 1
    if self.loss_type == 'ce':
        onehot = np.zeros((all_label_probs.shape))
        onehot[np.arange(all_label_probs.shape[0]), np.array(self.subset_labels)] = 1
        loss = -np.sum(onehot*np.log(label_probs+1e-6), axis=-1)
        entropy = -np.sum(label_probs*np.log(label_probs+1e-6), axis=-1)
        reward = self.previous_loss - self.entropy_coef * entropy - loss
        self.previous_loss = copy.deepcopy(loss)
    elif self.loss_type == 'step':
        predicts = np.argmax(label_probs, axis=-1)
        correct = (predicts == np.array(self.subset_labels)).astype(float)
        correct_probs = label_probs[np.arange(all_label_probs.shape[0]), np.array(self.subset_labels)]
        not_label_probs = torch.where(
            torch.Tensor(label_probs) == torch.Tensor(correct_probs).unsqueeze(1),
            torch.Tensor([-1]), torch.Tensor(label_probs))
        # [batch_size, num_classes]
        max_not_label_probs, _ = torch.max(not_label_probs, -1)
        # [batch_size, 1]

        # Compute piecewise gap reward
        gap = (torch.Tensor(correct_probs) - max_not_label_probs)
        correct = (gap > 0).long()
        step_reward = gap * (self.correct_bonus * correct + self.incorrect_bonus * (1 - correct))
        step_reward = step_reward.numpy()
        '''
        # print(correct_probs.shape, label_probs.shape, flush=True)
        prob_diff = np.expand_dims(correct_probs, axis=-1) - label_probs
        prob_diff = prob_diff + (prob_diff == 0).astype(float) * 1e6
        step_reward = np.min(prob_diff, axis=-1)
        step_reward = step_reward * correct * self.correct_bonus + (1 - correct) * step_reward * self.incorrect_bonus
        # equal = (np.sum(prob_diff**2, axis=-1) == 0).astype(float)
        # step_reward = step_reward * (1 - equal)
        '''
        reward = step_reward - self.previous_loss
        self.previous_loss = copy.deepcopy(step_reward)
        # assert np.mean(reward) < 10
    elif self.loss_type == 'acc':
        predicts = np.argmax(label_probs, axis=-1)
        entropy = -np.sum(label_probs*np.log(label_probs+1e-6), axis=-1)
        # if not self.evaluate:
        #     print(entropy, (predicts == np.array(self.subset_labels)))
        correct = (predicts == np.array(self.subset_labels)).astype(float) * 2 - 1 + self.entropy_coef * entropy
        reward = correct - self.previous_loss
        self.previous_loss = copy.deepcopy(correct)
        # print(reward, correct, prob_diff, flush=True)
        # predict_probs = label_probs[np.arange(all_label_probs.shape[0]), predicts]
        # reward = predict_probs - correct_probs
        # reward = correct * self.correct_bonus + (1 - correct) * reward * self.incorrect_bonus
    if self.loss_type == 'ce_sparse':
        onehot = np.zeros((all_label_probs.shape))
        onehot[np.arange(all_label_probs.shape[0]), np.array(self.subset_labels)] = 1
        loss = -np.sum(onehot*np.log(label_probs+1e-6), axis=-1)
        entropy = -np.sum(label_probs*np.log(label_probs+1e-6), axis=-1)
        reward = -loss
        self.previous_loss = copy.deepcopy(loss)
        if self.steps >= self.max_steps:
            reward = reward
        else:
            reward = reward * 0
    elif self.loss_type == 'step_sparse':
        predicts = np.argmax(label_probs, axis=-1)
        correct = (predicts == np.array(self.subset_labels)).astype(float)
        correct_probs = label_probs[np.arange(all_label_probs.shape[0]), np.array(self.subset_labels)]
        not_label_probs = torch.where(
            torch.Tensor(label_probs) == torch.Tensor(correct_probs).unsqueeze(1),
            torch.Tensor([-1]), torch.Tensor(label_probs))
        # [batch_size, num_classes]
        max_not_label_probs, _ = torch.max(not_label_probs, -1)
        # [batch_size, 1]

        # Compute piecewise gap reward
        gap = (torch.Tensor(correct_probs) - max_not_label_probs)
        correct = (gap > 0).long()
        step_reward = gap * (self.correct_bonus * correct + self.incorrect_bonus * (1 - correct))
        step_reward = step_reward.numpy()
        if self.steps >= self.max_steps:
            reward = step_reward
        else:
            reward = step_reward * 0
    elif self.loss_type == 'acc_sparse':
        predicts = np.argmax(label_probs, axis=-1)
        entropy = -np.sum(label_probs*np.log(label_probs+1e-6), axis=-1)
        correct = (predicts == np.array(self.subset_labels)).astype(float) * 2 - 1 + self.entropy_coef * entropy
        reward = correct
        if self.steps >= self.max_steps:
            reward = reward
        else:
            reward = reward * 0

    # Reward Scaling
    reward = reward * self.rew_scale
    # time2 = time.time()
    # print('reward ', time2 - time1)
    if self.steps >= self.max_steps:
        done = np.ones(self.subset_size)
        if self.evaluate:
            correct, total, tp, fp, fn = eval_accuracy(all_label_probs, self.subset_labels)
            info = {'episode_r': reward, 'correct': correct, 'total': total, 'orig_correct': self.orig_correct, 'orig_total': self.orig_total,
                    'tp': tp, 'fp': fp, 'fn': fn}
        else:
            info = {'episode_r': reward, 'step_r': step_reward * self.rew_scale}
    else:
        done = np.zeros(self.subset_size)
        info = {'episode_r': reward,  'step_r': step_reward * self.rew_scale}
    # print(action)
    # if self.evaluate:
    #     print('step: ', self.steps, ' current prompts: ', self.current_prompt_index)
    # done = np.array(self.terminate)
    # time_t2 = time.time()
    # print('total ', time_t2 - time_t1)
    return_obs = self.get_obs(obs, self.prev_actions)
    self.prev_actions = copy.deepcopy(np.array(action))
    return return_obs, reward, done, info

  def embedding(self, prompts, labels, verbalizer, sentences, sentences_verbalizer):
    # print('embedding')
    verbalized_prompt = self.verbalize(prompts, verbalizer)
    verbalized_sentences = self.verbalize(sentences, sentences_verbalizer, subset=True)
    all_embeddings = []
    for prompt_idx in range(len(verbalized_prompt[0])):
        prompt_subset = [[prompt[prompt_idx]] for prompt in verbalized_prompt]
        label_subset = [[label[prompt_idx]] for label in labels]
        _, obs = get_model_response_parallel(self.params, self.model, self.tokenizer, prompt_subset, label_subset, verbalized_sentences)
        all_embeddings.append(copy.deepcopy(obs))
    return_embeddings = []
    for sentence_idx in range(len(verbalized_sentences)):
        _return_embeddings = []
        for prompt_idx in range(len(verbalized_prompt[0])):
            _return_embeddings.append(all_embeddings[prompt_idx][sentence_idx])
        return_embeddings.append(_return_embeddings)
    return return_embeddings
    # all_embeddings = []
    # for _prompts, _labels, sentence in zip(verbalized_prompt, labels, verbalized_sentences):
    #     _all_embeddings = []
    #     for prompt, label in zip(_prompts, _labels):
    #         _, obs = get_model_response_parallel(self.params, [[prompt]], [[label]], [sentence])
    #         _all_embeddings.append(copy.deepcopy(obs))
    #     all_embeddings.append(_all_embeddings)
    # return all_embeddings

#   def knn(self, prompt_size, sentences, sentences_verbalizer, all_sentences, all_sentences_verbalizer, all_labels):
#     verbalized_all_sentences = self.verbalize(all_sentences, all_sentences_verbalizer, subset=True)
#     verbalized_sentences = self.verbalize(sentences, sentences_verbalizer, subset=True)
#     embeddings = []
#     for sentence in verbalized_sentences:
#         _, obs = get_model_response_parallel(self.params, self.model, self.tokenizer, [[""]], [[""]], [sentence])
#         embeddings.append(copy.deepcopy(obs))
#     embeddings = np.array(embeddings)
#     all_embeddings = []
#     for all_sentence in verbalized_all_sentences:
#         _, obs = get_model_response_parallel(self.params, self.model, self.tokenizer, [[""]], [[""]], [all_sentence])
#         all_embeddings.append(copy.deepcopy(obs))
#     all_embeddings = np.expand_dims(np.array(all_embeddings).squeeze(), axis=0)

#     # distance = np.sum((embeddings - all_embeddings)**2, axis=-1)
#     norm_embeddings = embeddings / np.sqrt(np.sum(embeddings**2, axis=-1, keepdims=True))
#     norm_all_embeddings = all_embeddings / np.sqrt(np.sum(all_embeddings**2, axis=-1, keepdims=True))
#     distance = np.sum(norm_embeddings * norm_all_embeddings, axis=-1)

#     # ind = np.argpartition(distance, -self.params['example_pool_size'], axis=-1)[:, :self.params['example_pool_size']]
#     ind = []
#     for row in range(distance.shape[0]):
#         ind.append(np.argsort(distance[row])[:self.params['example_pool_size']])
#     ind = np.array(ind)
    
#     knn_sentence_pool = []
#     knn_label_pool = []
#     add_knn_sentence_pool = []
#     add_knn_label_pool = []
#     for _ind in ind:
#         knn_sentence_pool.append([])
#         knn_label_pool.append([])
#         add_knn_sentence_pool.append([])
#         add_knn_label_pool.append([])
#         for __ind in _ind[:self.params['num_shots']]:
#             knn_sentence_pool[-1].append(all_sentences[__ind])
#             knn_label_pool[-1].append(all_labels[__ind])
#         for __ind in _ind[self.params['num_shots']:]:
#             add_knn_sentence_pool[-1].append(all_sentences[__ind])
#             add_knn_label_pool[-1].append(all_labels[__ind])
    
#     # print(len(knn_sentence_pool[0]), len(knn_label_pool[0]), len(add_knn_sentence_pool[0]))
#     return knn_sentence_pool, knn_label_pool, add_knn_sentence_pool, add_knn_label_pool

  def reset(self, idx=None, terminate=None):
    self.steps = 0
    if self.idxs is not None and self.evaluate:
        self.subset_size = self.idxs.shape[0]
        subset_idxs = self.idxs
    else:
        self.subset_size = self.num_processes
        # subset_idxs = np.random.permutation(len(self.train_sentences))[:self.subset_size]
        subset_idxs = np.random.choice(np.arange(len(self.train_sentences)), self.subset_size, replace=True)
    self.subset_idxs = subset_idxs
    self.terminate = [False for _ in range(self.subset_size)]
    self.prev_actions = np.array([0 for _ in range(self.subset_size)])
    self.subset_sentences = [copy.deepcopy(self.train_sentences[i]) for i in subset_idxs]
    # self.subset_sentences = self.train_sentences.select(subset_idxs)
    self.subset_labels = [copy.deepcopy(self.train_labels[i]) for i in subset_idxs]
    # Reset the verbalizer
    self.current_verbalizer = [[0 for _ in range(self.params['num_shots'])] for _ in range(len(self.subset_sentences))]
    self.add_current_verbalizer = [[0 for _ in range(self.params['example_pool_size'] - self.params['num_shots'])] for _ in range(len(self.subset_sentences))]
    self.deleted_verbalizer = [[] for _ in range(len(self.subset_sentences))]
    self.subset_verbalizer = [0 for _ in range(len(self.subset_sentences))]
    self.all_verbalizer = [0 for _ in range(len(self.all_prompt_sentence_pool))]
    
    # KNN select 
    if self.params['use_knn']:
        # print(self.prompt_sentence_pool, '\n')
        self.prompt_sentence_pool, self.prompt_label_pool, self.add_prompt_sentence_pool, self.add_prompt_label_pool = \
            self.knn(self.params['num_shots'], self.subset_sentences, self.subset_verbalizer, self.all_prompt_sentence_pool, self.all_verbalizer, self.all_prompt_label_pool)
        # First sample a batch of sentences
        self.current_prompt = copy.deepcopy(self.prompt_sentence_pool)
        self.current_prompt_labels = copy.deepcopy(self.prompt_label_pool)
        self.current_prompt_index = [np.arange(len(self.prompt_sentence_pool[0])) for _ in range(len(self.subset_sentences))]
        self.add_current_prompt = copy.deepcopy(self.add_prompt_sentence_pool)
        self.add_current_prompt_labels = copy.deepcopy(self.add_prompt_label_pool)
        self.add_current_prompt_index = [np.arange(len(self.add_prompt_sentence_pool[0]))+len(self.prompt_sentence_pool[0]) for _ in range(len(self.subset_sentences))]
        self.deleted_prompt = [[] for _ in range(len(self.subset_sentences))]
        self.deleted_prompt_labels = [[] for _ in range(len(self.subset_sentences))]
    else:
        # First sample a batch of sentences
        self.current_prompt = [copy.deepcopy(self.prompt_sentence_pool) for _ in range(len(self.subset_sentences))]
        self.current_prompt_labels = [copy.deepcopy(self.prompt_label_pool) for _ in range(len(self.subset_sentences))]
        self.current_prompt_index = [np.arange(len(self.prompt_sentence_pool)) for _ in range(len(self.subset_sentences))]
        self.add_current_prompt = [copy.deepcopy(self.add_prompt_sentence_pool) for _ in range(len(self.subset_sentences))]
        self.add_current_prompt_labels = [copy.deepcopy(self.add_prompt_label_pool) for _ in range(len(self.subset_sentences))]
        self.add_current_prompt_index = [np.arange(len(self.add_prompt_sentence_pool))+len(self.prompt_sentence_pool) for _ in range(len(self.subset_sentences))]
        self.deleted_prompt = [[] for _ in range(len(self.subset_sentences))]
        self.deleted_prompt_labels = [[] for _ in range(len(self.subset_sentences))]
    
    # Action history
    # self.act_histories = [[0 for _ in range(self.max_steps + 1)] for _ in range(len(self.subset_sentences))]
    # Prepare embeddings
    # self.current_prompt_embedding = self.embedding(self.current_prompt, self.current_prompt_labels, self.current_verbalizer, self.subset_sentences, self.subset_verbalizer)
    # self.add_current_prompt_embedding = self.embedding(self.add_current_prompt, self.add_current_prompt_labels, self.add_current_verbalizer, self.subset_sentences, self.subset_verbalizer)
    self.current_prompt_embedding = [copy.deepcopy(self.current_prompt_embedding_pool[i].numpy()) for i in subset_idxs]
    self.add_current_prompt_embedding = [copy.deepcopy(self.add_current_prompt_embedding_pool[i].numpy()) for i in subset_idxs]
    self.deleted_prompt_embedding = [[] for _ in range(len(self.subset_sentences))]
    # self.current_verbalizer_embedding = []
    # for verbalizer in range(len(self.prompt_template_keys)):
    #     self.current_verbalizer_embedding.append(self.embedding(self.current_prompt, self.current_prompt_labels, (np.array(self.current_verbalizer)+verbalizer).tolist(), self.subset_sentences, self.subset_verbalizer))
    # self.current_verbalizer_embedding = np.transpose(np.array(self.current_verbalizer_embedding), axes=(1, 2, 0, 3)).tolist()
    # self.add_current_verbalizer_embedding = []
    # for verbalizer in range(len(self.prompt_template_keys)):
    #     self.add_current_verbalizer_embedding.append(self.embedding(self.add_current_prompt, self.add_current_prompt_labels, (np.array(self.add_current_verbalizer)+verbalizer).tolist(), self.subset_sentences, self.subset_verbalizer))
    # self.add_current_verbalizer_embedding = np.transpose(np.array(self.add_current_verbalizer_embedding), axes=(1, 2, 0, 3)).tolist()
    self.current_verbalizer_embedding = [copy.deepcopy(self.current_verbalizer_embedding_pool[i].numpy()) for i in subset_idxs]
    self.add_current_verbalizer_embedding = [copy.deepcopy(self.add_current_verbalizer_embedding_pool[i].numpy()) for i in subset_idxs]
    # print(len(self.add_current_verbalizer_embedding), len(self.add_current_verbalizer_embedding[0]), len(self.add_current_verbalizer_embedding[0][0]))
    self.deleted_verbalizer_embedding = [[] for _ in range(len(self.subset_sentences))]
    # print(np.array(self.current_verbalizer_embedding).shape, np.array(self.add_current_verbalizer_embedding).shape)

    #TODO: changes here
    if not self.evaluate and self.params['random_init'] > 0:
        for i in range(self.subset_size):
            idxs = np.random.permutation(self.params['example_pool_size'])
            all_prompt = self.current_prompt[i] + self.add_current_prompt[i]
            all_prompt_label = self.current_prompt_labels[i] + self.add_current_prompt_labels[i]
            all_prompt_index = self.current_prompt_index[i].tolist() + self.add_current_prompt_index[i].tolist()
            self.current_prompt[i] = [copy.deepcopy(all_prompt[idx]) for idx in idxs[:self.params['num_shots']]]
            self.current_prompt_labels[i] = [copy.deepcopy(all_prompt_label[idx]) for idx in idxs[:self.params['num_shots']]]
            self.current_prompt_index[i] = [copy.deepcopy(all_prompt_index[idx]) for idx in idxs[:self.params['num_shots']]]
            self.add_current_prompt[i] = [copy.deepcopy(all_prompt[idx]) for idx in idxs[self.params['num_shots']:]]
            self.add_current_prompt_labels[i] = [copy.deepcopy(all_prompt_label[idx]) for idx in idxs[self.params['num_shots']:]]
            self.add_current_prompt_index[i] = [copy.deepcopy(all_prompt_index[idx]) for idx in idxs[self.params['num_shots']:]]

            all_prompt_embedding = np.concatenate([np.array(self.current_prompt_embedding[i]), np.array(self.add_current_prompt_embedding[i])], axis=0)
            all_verbalizer_embedding = np.concatenate([np.array(self.current_verbalizer_embedding[i]), np.array(self.add_current_verbalizer_embedding[i])], axis=0)
            self.current_prompt_embedding[i] = [copy.deepcopy(all_prompt_embedding[idx]) for idx in idxs[:self.params['num_shots']]]
            self.add_current_prompt_embedding[i] = [copy.deepcopy(all_prompt_embedding[idx]) for idx in idxs[self.params['num_shots']:]]
            self.current_verbalizer_embedding[i] = [copy.deepcopy(all_verbalizer_embedding[idx]) for idx in idxs[:self.params['num_shots']]]
            self.add_current_verbalizer_embedding[i] = [copy.deepcopy(all_verbalizer_embedding[idx]) for idx in idxs[self.params['num_shots']:]]

            if self.params['random_init'] > 1:
                self.current_verbalizer[i] = np.random.randint(len(self.prompt_template_keys), size=self.params['num_shots']).tolist()
            if self.params['random_init'] > 2:
                self.add_current_verbalizer[i] = np.random.randint(len(self.prompt_template_keys), size=self.params['example_pool_size'] - self.params['num_shots']).tolist()

    if self.verbalizer:
        verbalized_prompt = self.verbalize(self.current_prompt, self.current_verbalizer)
        verbalized_pool = self.verbalize(self.add_current_prompt, self.add_current_verbalizer)
        subset_sentences = self.verbalize(self.subset_sentences, self.subset_verbalizer, subset=True)
    else:
        verbalized_prompt = self.current_prompt
        verbalized_pool = self.add_prompt_sentence_pool
        subset_sentences = self.subset_sentences
    raw_resp, obs = get_model_response_parallel(self.params, self.model, self.tokenizer, verbalized_prompt, self.current_prompt_labels, subset_sentences)
    # _, add_obs = get_model_response_parallel(self.params, verbalized_pool, self.add_current_prompt_labels, subset_sentences)
    # obs = np.concatenate([obs, add_obs], axis=-1)
    all_label_probs = get_label_probs(self.params, raw_resp, verbalized_prompt, self.current_prompt_labels, subset_sentences)
    # print('reset ', self.idxs, ' ', all_label_probs)

    if self.evaluate:
        self.orig_correct, self.orig_total, _, _, _ = eval_accuracy(all_label_probs, self.subset_labels)

    assert len(all_label_probs) == len(self.subset_labels)
    # label_probs = all_label_probs / (np.sum(all_label_probs, axis=-1, keepdims=True) + (np.sum(all_label_probs, axis=-1, keepdims=True) == 0))
    label_probs = all_label_probs / np.sum(all_label_probs, axis=-1, keepdims=True)
    if self.loss_type == 'ce':
        onehot = np.zeros((all_label_probs.shape))
        onehot[np.arange(all_label_probs.shape[0]), np.array(self.subset_labels)] = 1
        loss = -np.sum(onehot*np.log(label_probs+1e-6), axis=-1)
        entropy = -np.sum(label_probs*np.log(label_probs+1e-6), axis=-1)
        self.previous_loss = copy.deepcopy(loss) - self.entropy_coef * entropy
    elif self.loss_type == 'step':
        predicts = np.argmax(label_probs, axis=-1)
        correct = (predicts == np.array(self.subset_labels)).astype(float)
        correct_probs = label_probs[np.arange(all_label_probs.shape[0]), np.array(self.subset_labels)]
        not_label_probs = torch.where(
            torch.Tensor(label_probs) == torch.Tensor(correct_probs).unsqueeze(1),
            torch.Tensor([-1]), torch.Tensor(label_probs))
        # [batch_size, num_classes]
        max_not_label_probs, _ = torch.max(not_label_probs, -1)
        # [batch_size, 1]

        # Compute piecewise gap reward
        gap = (torch.Tensor(correct_probs) - max_not_label_probs)
        correct = (gap > 0).long()
        step_reward = gap * (self.correct_bonus * correct + self.incorrect_bonus * (1 - correct))
        step_reward = step_reward.numpy()
        '''
        # print(correct_probs.shape, label_probs.shape, flush=True)
        prob_diff = np.expand_dims(correct_probs, axis=-1) - label_probs
        prob_diff = prob_diff + (prob_diff == 0).astype(float) * 1e6
        step_reward = np.min(prob_diff, axis=-1)
        step_reward = correct * step_reward * self.correct_bonus + (1 - correct) * step_reward * self.incorrect_bonus
        '''
        self.previous_loss = copy.deepcopy(step_reward)
        # print(reward, correct, prob_diff, flush=True)
        # predict_probs = label_probs[np.arange(all_label_probs.shape[0]), predicts]
        # reward = predict_probs - correct_probs
        # reward = correct * self.correct_bonus + (1 - correct) * reward * self.incorrect_bonus
    elif self.loss_type == 'acc':
        predicts = np.argmax(label_probs, axis=-1)
        entropy = -np.sum(label_probs*np.log(label_probs+1e-6), axis=-1)
        # if not self.evaluate:
        #     print(entropy, (predicts == np.array(self.subset_labels)))
        correct = (predicts == np.array(self.subset_labels)).astype(float) * 2 - 1 + self.entropy_coef * entropy
        self.previous_loss = copy.deepcopy(correct)
    elif self.loss_type == 'step_sparse':
        predicts = np.argmax(label_probs, axis=-1)
        correct = (predicts == np.array(self.subset_labels)).astype(float)
        correct_probs = label_probs[np.arange(all_label_probs.shape[0]), np.array(self.subset_labels)]
        not_label_probs = torch.where(
            torch.Tensor(label_probs) == torch.Tensor(correct_probs).unsqueeze(1),
            torch.Tensor([-1]), torch.Tensor(label_probs))
        # [batch_size, num_classes]
        max_not_label_probs, _ = torch.max(not_label_probs, -1)
        # [batch_size, 1]

        # Compute piecewise gap reward
        gap = (torch.Tensor(correct_probs) - max_not_label_probs)
        correct = (gap > 0).long()
        step_reward = gap * (self.correct_bonus * correct + self.incorrect_bonus * (1 - correct))
        step_reward = step_reward.numpy()
        self.previous_loss = copy.deepcopy(step_reward)
        # print(reward, correct, prob_diff, flush=True)
        # predict_probs = label_probs[np.arange(all_label_probs.shape[0]), predicts]
        # reward = predict_probs - correct_probs
        # reward = correct * self.correct_bonus + (1 - correct) * reward * self.incorrect_bonus

    return self.get_obs(obs, self.prev_actions)

