import numpy as np
import time
from copy import deepcopy
import os
import sys
import torch
import pickle
import openai
from transformers import GPT2Tokenizer, GPT2LMHeadModel, RobertaConfig, RobertaTokenizer, RobertaForMaskedLM

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    bs = params['bs']
    if bs is None:
        if 'gpt2' in params['model']:
            return 1
        else:
            assert params['model'] in ['ada', 'babbage', 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta', 'davinci-beta']
            return 20
    else:
        return bs

def random_sampling(sentences, labels, num):
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    return deepcopy(selected_sentences), deepcopy(selected_labels)

def setup_roberta(model_name, gpu_id):
    config = RobertaConfig.from_pretrained(model_name)
    roberta_tokenizer = RobertaTokenizer.from_pretrained(model_name)
    roberta_model = RobertaForMaskedLM.from_pretrained(model_name, config=config)
    roberta_model.eval().to('cuda:'+str(gpu_id))

    print("Finished Model Setup")
    return roberta_model, roberta_tokenizer

def complete_roberta(params, roberta_model, roberta_tokenizer, prompt, l=10, model_name='roberta-large', num_log_probs=None, echo=False, verbalizer=None):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    # rank = torch.distributed.get_rank()
    # local_rank = int(os.getenv('LOCAL_RANK', '0'))
    # world_size = int(os.getenv('WORLD_SIZE', '1'))
    if isinstance(prompt, str):
        prompt = [prompt] # the code below assumes a list
    for prompt_idx in range(len(prompt)):
        prompt[prompt_idx] += ' <mask>'
    if verbalizer is None:
        label_idx = [roberta_tokenizer.convert_tokens_to_ids('\u0120'+label) for label in params['inv_label_dict'].keys()]
    else:
        label_idx = [roberta_tokenizer.convert_tokens_to_ids('\u0120'+label) for label in verbalizer[0].keys()]
    input_ids = roberta_tokenizer.batch_encode_plus(prompt, return_tensors="pt").to(roberta_model.device)

    if input_ids['input_ids'].shape[1] > 512:
        input_ids['input_ids'] = input_ids['input_ids'][:, -512:]
        input_ids['attention_mask'] = input_ids['attention_mask'][:, -512:]
    mask_ids = (input_ids['input_ids'] == 50264)[0].nonzero(as_tuple=True)[0]
    
    # they want the probs of the top tokens
    if num_log_probs is not None:
        # we are left padding, so we need to adjust the position IDs
        attention_mask = (input_ids['input_ids'] != 50256).float()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        outputs_list = []
        with torch.no_grad():
            outputs = roberta_model(**input_ids, output_hidden_states=True)

        logits = outputs.logits.detach().cpu()
        embeddings = outputs.hidden_states[-1][:, mask_ids[0]].detach().cpu().numpy()
        if not echo:
            # get the top tokens and probs for the generated l tokens
            probs = torch.softmax(logits[:, mask_ids].float(), dim=2).cpu()
        else:
            assert False
            # get the top tokens and probs for the context and the generated l tokens
            probs = torch.softmax(logits.float(), dim=2).cpu()
        top_tokens = torch.cat([torch.tensor(np.array(label_idx)).to(probs.device).unsqueeze(0).unsqueeze(0) for _ in range(probs.shape[0])], dim=0)
        top_probs = probs[:, :, torch.tensor(np.array(label_idx)).to(probs.device)]

        logprobs = torch.log(probs)
        top_log_probs = torch.log(top_probs)

    total_sequences = input_ids['input_ids']
    # create the return value to resemble OpenAI
    return_json = {}
    choices = []
    for batch_id in range(len(prompt)):
        curr_json = {}
        # text is just the optional context and next l tokens
        if not echo:
            curr_json['text'] = roberta_tokenizer.decode(total_sequences[batch_id][mask_ids], skip_special_tokens=True)
        else:
            assert False
            curr_json['text'] = roberta_tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

        # fill the return json with the top tokens and probs to match the OpenAI return value.
        if num_log_probs is not None:
            curr_json['logprobs'] = {}
            curr_json['logprobs']['top_logprobs'] = []
            curr_json['logprobs']['token_logprobs'] = []
            curr_json['logprobs']['tokens'] = []
            if not echo:
                # cutoff the -1 here because the probs are shifted one over for LMs
                for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id], top_tokens[batch_id]):
                    # tokens is a list of the top token at each position
                    curr_json['logprobs']['tokens'].append(roberta_tokenizer.decode([current_element_top_tokens[0]]))
                    # token_logprobs is a list of the logprob of the top token at each position
                    curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                    # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[roberta_tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
            else:
                assert False
                # same as not above but small tweaks
                # we add null to the front because for the GPT models, they have null probability for the first token
                # (for some reason they don't have an beginning of sentence token)
                curr_json['logprobs']['top_logprobs'].append('null')
                # cutoff the -1 here because the probs are shifted one over for LMs
                for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1])):
                    # skip padding tokens
                    if total_sequences[batch_id][index].item() == 50256:
                        continue
                    temp = {}
                    for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                        temp[roberta_tokenizer.decode(token.item())] = log_prob.item()
                    curr_json['logprobs']['top_logprobs'].append(temp)
                for index in range(len(probs[batch_id])):
                    curr_json['logprobs']['tokens'].append(roberta_tokenizer.decode([total_sequences[batch_id][index]]))
                curr_json['logprobs']['token_logprobs'].append('null')
                for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
                    # probs are left shifted for LMs 
                    curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index+1]])

        choices.append(curr_json)

    return_json['choices'] = choices
    return return_json, embeddings

def complete(params, model, tokenizer, prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None, verbalizer=None):
    """complete the prompt using a language model"""
    assert l >= 0
    assert temp >= 0
    if 'roberta' in model_name:
        assert n == None # unsupported at the moment
        assert temp == 0 # unsupported at the moment
        # setup_roberta(model)
        return complete_roberta(params, model, tokenizer, prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, verbalizer=verbalizer)

def construct_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            assert isinstance(l, str) # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    return prompt

def get_model_response(params, train_sentences, train_labels, test_sentences, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None):
    """
    Obtain model's responses on test sentences, given the training examples
    :param params: parameters for the experiment
    :param train_sentences: few-shot training sentences
    :param train_labels: few-shot training labels
    :param test_sentences: few-shot test sentences
    :param return_all_prompts: whether to return all the prompts
    :param num_tokens_to_predict_override: whether to override num token to predict
    :param override_prompt: whether to override prompt
    :return: a list of dictionaries
    """
    all_raw_answers = []

    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    if override_prompt is None:
        prompts = []
        for test_sentence in test_sentences:
            prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))
    else:
        prompts = override_prompt

    chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        if num_tokens_to_predict_override is not None:
            num_tokens_to_predict = num_tokens_to_predict_override
        else:
            num_tokens_to_predict = params['num_tokens_to_predict']
        resp = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'])
        for answer_id, answer in enumerate(resp['choices']):
            all_raw_answers.append(answer)

    if return_all_prompts:
        return all_raw_answers, prompts
    else:
        return all_raw_answers

def construct_prompt_parallel(params, train_sentences, train_labels, test_sentence, verbalizers=None, prefix=None):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    # if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
    #     return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    if prefix is None:
        prompt = params["prompt_prefix"]
    else:
        prompt = prefix
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    if verbalizers is None:
        for s, l in zip(train_sentences, train_labels):
            if s == "" and l == "":
                continue
            prompt += q_prefix
            prompt += s + "\n"
            if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
                # assert params['task_format'] == 'classification'
                l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
            else:
                assert isinstance(l, str) # string labels
                # assert params['task_format'] == 'qa'
                l_str = l

            prompt += a_prefix
            prompt += l_str + "\n\n"
    else:
        for s, l, v in zip(train_sentences, train_labels, verbalizers):
            if s == "" and l == "":
                continue
            prompt += q_prefix
            prompt += s + "\n"
            if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
                # assert params['task_format'] == 'classification'
                # l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
                l_str = list(v.keys())[l]
            else:
                assert isinstance(l, str) # string labels
                # assert params['task_format'] == 'qa'
                l_str = l

            prompt += a_prefix
            prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    return prompt

def get_model_response_parallel(params, model, tokenizer, train_sentences, train_labels, test_sentences, prefix=None, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None, verbalizer=None, prompt_verbalizer=None):
    """
    Obtain model's responses on test sentences, given the training examples
    :param params: parameters for the experiment
    :param train_sentences: few-shot training sentences
    :param train_labels: few-shot training labels
    :param test_sentences: few-shot test sentences
    :param return_all_prompts: whether to return all the prompts
    :param num_tokens_to_predict_override: whether to override num token to predict
    :param override_prompt: whether to override prompt
    :return: a list of dictionaries
    """
    all_raw_answers = []
    all_embeddings = []

    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    if override_prompt is None:
        prompts = []
        if prompt_verbalizer is not None:
            if prefix is not None:
                for _train_sentences, _train_labels, _prompt_verbalizer, test_sentence, _prefix in zip(train_sentences, train_labels, prompt_verbalizer, test_sentences, prefix):
                    prompts.append(construct_prompt_parallel(params, _train_sentences, _train_labels, test_sentence, prefix=_prefix, verbalizers=_prompt_verbalizer))
            else:
                for _train_sentences, _train_labels, _prompt_verbalizer, test_sentence in zip(train_sentences, train_labels, prompt_verbalizer, test_sentences):
                    prompts.append(construct_prompt_parallel(params, _train_sentences, _train_labels, test_sentence, verbalizers=_prompt_verbalizer))
        else:
            if prefix is not None:
                for _train_sentences, _train_labels, test_sentence, _prefix in zip(train_sentences, train_labels, test_sentences, prefix):
                    prompts.append(construct_prompt_parallel(params, _train_sentences, _train_labels, test_sentence, prefix=_prefix))
            else:
                for _train_sentences, _train_labels, test_sentence in zip(train_sentences, train_labels, test_sentences):
                    prompts.append(construct_prompt_parallel(params, _train_sentences, _train_labels, test_sentence))
    else:
        prompts = override_prompt

    if verbalizer is not None:
        chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
        chunked_verbalizers = list(chunks(verbalizer, chunk_size_helper(params)))
        for chunk_id, (test_chunk_prompts, test_chunk_verbalizers) in enumerate(zip(chunked_prompts, chunked_verbalizers)):
            if num_tokens_to_predict_override is not None:
                num_tokens_to_predict = num_tokens_to_predict_override
            else:
                num_tokens_to_predict = params['num_tokens_to_predict']
            # TODO: remove the top k log probs
            resp, embeddings = complete(params, model, tokenizer, test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'], verbalizer=test_chunk_verbalizers)
            # resp, embeddings = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=1000)
            for answer_id, answer in enumerate(resp['choices']):
                all_raw_answers.append(answer)
            all_embeddings.append(embeddings.astype(float))
    else:
        chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
        for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
            if num_tokens_to_predict_override is not None:
                num_tokens_to_predict = num_tokens_to_predict_override
            else:
                num_tokens_to_predict = params['num_tokens_to_predict']
            # TODO: remove the top k log probs
            resp, embeddings = complete(params, model, tokenizer, test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'])
            # resp, embeddings = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=1000)
            for answer_id, answer in enumerate(resp['choices']):
                all_raw_answers.append(answer)
            all_embeddings.append(embeddings.astype(float))

    if return_all_prompts:
        return all_raw_answers, np.concatenate(all_embeddings, axis=0), prompts
    else:
        return all_raw_answers, np.concatenate(all_embeddings, axis=0)

def load_pickle(params):
    # load saved results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data

def save_pickle(params, data):
    # save results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data

def print_results(tree, names=('Original Accuracy  ','Calibrated Accuracy')):
    # print out all results
    root = deepcopy(tree)
    for dataset in root.keys():
        print(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            print(f"\nModel: {model}")
            num_shots_node = models_node[model]
            for num_shots in num_shots_node.keys():
                accuracies = np.array(list(num_shots_node[num_shots].values()))
                accuracies_mean = np.mean(accuracies, axis=0)
                accuracies_low = np.min(accuracies, axis=0)
                accuracies_high = np.max(accuracies, axis=0)
                accuracies_std = np.std(accuracies, axis=0)

                print(f"\n{num_shots}-shot, {len(accuracies)} seeds")
                for i, (m, l, h, s) in enumerate(zip(accuracies_mean, accuracies_low, accuracies_high, accuracies_std)):
                    print(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")
                print()

def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = saved_result['accuracies']
    print_results(result_tree)
