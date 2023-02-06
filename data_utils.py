import pandas as pd
import json
import pickle
import numpy as np
from utils import ROOT_DIR

def load_sst2():
    from datasets import load_dataset
    train_sentences = load_dataset('glue', 'sst2', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('glue', 'sst2', split='validation')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels

def load_qnli():
    from datasets import load_dataset
    train_sentences = load_dataset('glue', 'qnli', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('glue', 'qnli', split='validation')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels

def load_mnli():
    from datasets import load_dataset
    train_sentences = load_dataset('glue', 'mnli', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('glue', 'mnli', split='validation_matched')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels

def load_mrpc():
    from datasets import load_dataset
    train_sentences = load_dataset('glue', 'mrpc', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('glue', 'mrpc', split='validation')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels

def load_customer_review():
    from datasets import load_dataset
    file_dict = {'train': 'cr/16-42/train.tsv'}
    train_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    train_labels = train_sentences['label']
    file_dict = {'train': 'cr/test.tsv'}
    test_sentences = load_dataset('csv', data_files=file_dict, split='train', delimiter='\t')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels

def load_agnews():
    from datasets import load_dataset
    train_sentences = load_dataset('ag_news', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('ag_news', split='test')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences] 
    return train_sentences, train_labels, test_sentences, test_labels

def load_trec():
    from datasets import load_dataset
    train_sentences = load_dataset('trec', split='train')
    train_labels = train_sentences['label-coarse']
    test_sentences = load_dataset('trec', split='test')
    test_labels = test_sentences['label-coarse']
    str2int = train_sentences.features['label-coarse']._str2int
    int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels, str2int, int2str

def load_yelp_polarity():
    from datasets import load_dataset
    train_sentences = load_dataset('yelp_polarity', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('yelp_polarity', split='test')
    test_labels = test_sentences['label']
    str2int = train_sentences.features['label']._str2int
    int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels, str2int, int2str

def load_rte():
    from datasets import load_dataset
    train_sentences = load_dataset('super_glue', 'rte', split='train')
    train_labels = train_sentences['label']
    unique = {label: idx for idx, label in enumerate(set(train_labels))}
    train_labels = [unique[label] for label in train_sentences['label']]
    test_sentences = load_dataset('super_glue', 'rte', split='validation')
    test_labels = test_sentences['label']
    unique = {label: idx for idx, label in enumerate(set(test_labels))}
    test_labels = [unique[label] for label in test_sentences['label']]
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]

    return train_sentences, train_labels, test_sentences, test_labels

def load_rotten_tomatoes():
    from datasets import load_dataset
    train_sentences = load_dataset('rotten_tomatoes', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('rotten_tomatoes', split='test')
    test_labels = test_sentences['label']
    str2int = train_sentences.features['label']._str2int
    int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]

    return train_sentences, train_labels, test_sentences, test_labels, str2int, int2str

def load_snli():
    from datasets import load_dataset
    train_sentences = load_dataset('snli', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('snli', split='validation')
    test_labels = test_sentences['label']
    str2int = train_sentences.features['label']._str2int
    int2str = inv_map = {v: k for k, v in str2int.items()}
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]

    return train_sentences, train_labels, test_sentences, test_labels, str2int, int2str

def load_dbpedia():
    from datasets import load_dataset
    train_sentences = load_dataset('dbpedia_14', split='train')
    train_labels = train_sentences['label']
    test_sentences = load_dataset('dbpedia_14', split='test')
    test_labels = test_sentences['label']
    train_sentences = [sentence for sentence in train_sentences]
    test_sentences = [sentence for sentence in test_sentences]
    return train_sentences, train_labels, test_sentences, test_labels

def load_slot_movies(field_name):
    all_fields = ["Actor", "Award", "Character_Name", "Director", "Genre", "Opinion", "Origin", "Plot", "Quote", "Relationship", "Soundtrack", "Year"]
    assert field_name in all_fields
    all_fields.remove(field_name)
    filter_tags = [f"B-{field}" for field in all_fields] + [f"I-{field}" for field in all_fields] + ["O"]
    target_tags = [f"B-{field_name}", f"I-{field_name}"]

    with open(f'{ROOT_DIR}/data/slot-movies/train', 'r') as f:
        lines = f.readlines()
        lines = [line.replace(' <=> <NULL>','').strip() for line in lines]
    train_answers = []
    train_sentences = []
    for line in lines:
        answer = ''
        untagged_line = ''
        for word in line.split(' '):
            contains_target = [tag in word for tag in target_tags]
            if np.any(contains_target):
                for tag in target_tags:
                    word = word.replace(':' + tag, '')
                answer += word + ' '
            for tag in filter_tags:
                word = word.replace(':' + tag, '')
            untagged_line += word + ' '

        if answer != '':
            train_answers.append(answer.strip())
            train_sentences.append(untagged_line.strip())

    with open(f'{ROOT_DIR}/data/slot-movies/test', 'r') as f:
        lines = f.readlines()
        lines = [line.replace(' <=> <NULL>','').strip() for line in lines]
    test_answers = []
    test_sentences = []
    for line in lines:
        answer = ''
        untagged_line = ''
        for word in line.split(' '):
            contains_target = [tag in word for tag in target_tags]
            if np.any(contains_target):
                for tag in target_tags:
                    word = word.replace(':' + tag, '')
                answer += word + ' '
            for tag in filter_tags:
                word = word.replace(':' + tag, '')
            untagged_line += word + ' '

        if answer != '':
            test_answers.append(answer.strip())
            test_sentences.append(untagged_line.strip())

    return train_sentences, train_answers, test_sentences, test_answers

def load_atis(tag_name):
    with open(f'{ROOT_DIR}/data/atis/atis.train.pkl', 'rb') as stream:
        ds,dicts = pickle.load(stream)

    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])
    query, slots, intent =  map(ds.get, ['query', 'slot_labels', 'intent_labels'])

    tags_dict = {}
    train_sentences = []
    train_slot_strings = []
    for i in range(len(query)):
        slot_string = ''
        beginning_count = 0 # when there are multiple mentions of the destination city, we want to avoid those
        for j in range(len(query[i])):
            tag = i2s[slots[i][j]][2:]
            if tag in tags_dict.keys():
                tags_dict[tag] += 1
            else:
                tags_dict[tag] = 1

            if f'B-{tag_name}' in i2s[slots[i][j]]:
                beginning_count += 1
            if tag_name in i2s[slots[i][j]]:
                slot_string += i2t[query[i][j]] + ' '
        if slot_string != '' and beginning_count == 1:
            train_sentences.append(' '.join(map(i2t.get, query[i][1:-1]))) # [1:-1] cuts off BOS and EOS
            train_slot_strings.append(slot_string.strip())

    with open(f'{ROOT_DIR}/data/atis/atis.test.pkl', 'rb') as stream:
        ds,dicts = pickle.load(stream)

    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])
    query, slots, intent =  map(ds.get, ['query', 'slot_labels', 'intent_labels'])

    test_sentences = []
    test_slot_strings = []
    for i in range(len(query)):
        slot_string = ''
        beginning_count = 0 # when there are multiple mentions of the destination city, we want to avoid those
        for j in range(len(query[i])):
            if f'B-{tag_name}' in i2s[slots[i][j]]:
                beginning_count += 1
            if tag_name in i2s[slots[i][j]]:
                slot_string += i2t[query[i][j]] + ' '
        if slot_string != '' and beginning_count == 1:
            test_sentences.append(' '.join(map(i2t.get, query[i][1:-1]))) # [1:-1] cuts off BOS and EOS
            test_slot_strings.append(slot_string.strip())

    return train_sentences, train_slot_strings, test_sentences, test_slot_strings

def load_lama(which_lama):
    ### Load test data
    with open(f'{ROOT_DIR}/data/lama/original_rob/P{which_lama}/test.jsonl', 'r') as json_file:
        json_list = list(json_file)
    all_y_test = []
    all_x_test = []
    for json_str in json_list:
        result = json.loads(json_str)
        all_y_test.append(result['obj_label'])
        all_x_test.append(result['sub_label'])

    ### Load train data
    with open(f'{ROOT_DIR}/data/lama/original_rob/P{which_lama}/train.jsonl', 'r') as json_file:
        json_list = list(json_file)
    all_y_train = []
    all_x_train = []
    for json_str in json_list[:1000]:
        result = json.loads(json_str)
        all_y_train.append(result['obj_label'])
        all_x_train.append(result['sub_label'])

    with open(f'{ROOT_DIR}/data/lama/relations.jsonl', 'r') as json_file:
        json_list = list(json_file)
    template = None
    for json_str in json_list:
        result = json.loads(json_str)
        idx = int(result['relation'][1:])
        if idx == which_lama:
            template = result['template']
            x_pos = template.find('[X]')
            y_pos = template.find('[Y]')
            assert (x_pos >= 0) and (y_pos >= 0), "placeholder not found"
            if x_pos > y_pos:
                print("Not auto-regressive, skip")
                template = "INVALID"
            break

    return all_x_train, all_y_train, all_x_test, all_y_test, template

def custom_load_dataset(params, change_params=True):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params['dataset'] == 'glue/sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative.\n\n"
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['terrible'], 1: ['great']}
            params['inv_label_dict'] = {'terrible': 0, 'great': 1}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1
    
    elif params['dataset'] == 'glue/qnli':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_qnli()
        if change_params:
            params['prompt_prefix'] = "You are given two sentences(Sentence1 and Sentence2). Answer \"yes\" if these sentences are a paraphrase of one another, otherwise answer \"no\".\n\n"
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['no'], 1: ['yes']}
            params['inv_label_dict'] = {'no': 0, 'yes': 1}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'glue/mnli':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mnli()
        if change_params:
            params['prompt_prefix'] = "In this task, you're given a pair of sentences, sentence 1 and sentence 2. Your job is to choose whether the two sentences clearly agree (entailment)/disagree (contradiction) with each other, or if this cannot be determined (neutral). Your answer must be in the form of the letters Yes, Maybe, and No respectively.\n\n"
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['Yes'], 1: ['Maybe'], 2: ['No']}
            params['inv_label_dict'] = {'Yes': 0, 'Maybe': 1, 'No': 2}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'glue/mrpc':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_mrpc()
        if change_params:
            params['prompt_prefix'] = "You are given two sentences(Sentence1 and Sentence2). Answer \"Yes\" if these sentences are a paraphrase of one another, otherwise answer \"No\".\n\n"
            params["q_prefix"] = " "
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
            params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'customer_review':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_customer_review()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences from customer reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative.\n\n"
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['terrible'], 1: ['great']}
            params['inv_label_dict'] = {'terrible': 0, 'great': 1}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'ag_news':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        if change_params:
            params['prompt_prefix'] = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
            params["q_prefix"] = "Article: "
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology']}
            params['inv_label_dict'] = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3} # notice index start from 1 here
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'trec':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels, str2int, int2str = load_trec()
        if change_params:
            params['prompt_prefix'] = ""
            params["q_prefix"] = " "
            params["a_prefix"] = " "
            # New task dict
            params['label_dict'] = {0: ['Description'], 1: ['Entity'], 2: ['Ab'], 3: ['Description'], 4: ['Number'], 5: ['Location']}
            params['inv_label_dict'] = {'Description': 0, 'Entity': 1, 'Ab': 2, 'Description': 3, 'Number': 4, 'Location': 5}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'super_glue/rte':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_rte()
        if change_params:
            params['prompt_prefix'] = ""
            params["q_prefix"] = " "
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['Yes'], 1: ['No']}
            params['inv_label_dict'] = {'Yes': 0, 'No': 1}
            params['num_user_input'] = 2
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'yelp_polarity':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels, str2int, int2str = load_yelp_polarity()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences from Yelp reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative.\n\n"
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['terrible'], 1: ['great']}
            params['inv_label_dict'] = {'terrible': 0, 'great': 1}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'rotten_tomatoes':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels, str2int, int2str = load_rotten_tomatoes()
        if change_params:
            params['prompt_prefix'] = "In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative.\n\n"
            params["q_prefix"] = "Review: "
            params["a_prefix"] = "Sentiment: "
            params['label_dict'] = {0: ['terrible'], 1: ['great']}
            params['inv_label_dict'] = {'terrible': 0, 'great': 1}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'snli':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels, str2int, int2str = load_snli()
        if change_params:
            params['prompt_prefix'] = "In this task, you're given a pair of sentences, sentence 1 and sentence 2. Your job is to choose whether the two sentences clearly agree (entailment)/disagree (contradiction) with each other, or if this cannot be determined (neutral). Your answer must be in the form of the letters Yes, Maybe, and No respectively.\n\n"
            params["q_prefix"] = " "
            params["a_prefix"] = "Answer: "
            params['label_dict'] = {0: ['Yes'], 1: ['Maybe'], 2: ['No']}
            params['inv_label_dict'] = {'Yes': 0, 'Maybe': 1, 'No': 2}
            params['task_format'] = 'classification'
            params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'cb':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = get_cb()
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['false'], 1: ['neither'], 2: ['true']}
        params['inv_label_dict'] = {'false': 0, 'neither': 1, 'true': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'] == 'dbpedia_14':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia()
        params['prompt_prefix'] = "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Athlete'], 4: ['Polit'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}
        params['inv_label_dict'] = {'Company': 0, 'School': 1, 'Artist': 2, 'Athlete': 3, 'Polit': 4, 'Transportation': 5, 'Building': 6, 'Nature': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'Book': 13}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['dataset'][:4] == 'lama':
        which_lama = int(params['dataset'].split('_')[-1])
        all_x_train, all_y_train, all_x_test, all_y_test, template = load_lama(which_lama)

        # reject if template is not valid
        if template == "INVALID":
            params['template'] = template
            return None, None, None, None

        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = all_x_train, all_y_train, all_x_test, all_y_test
        params['prompt_prefix'] = ""
        params['task_format'] = 'qa'
        params['num_tokens_to_predict'] = 1
        params['template'] = template

        x_pos = template.find('[X]')
        y_pos = template.find('[Y]')
        seg1 = template[0:x_pos]
        seg2 = template[x_pos+3:y_pos]

        def single_prompt_func(entity, target):
            return f"{seg1}{entity}{seg2}{target}"

        def prompt_func(params, train_sentences, train_labels, test_sentence, test_label_option=None):
            assert seg2[-1] == " "
            prompt = ""
            for x, y in zip(train_sentences, train_labels):
                prompt += single_prompt_func(x, y)
                prompt += "\n\n"

            if test_label_option is None:
                prompt += f"{seg1}{test_sentence}{seg2}"[:-1]
            else:
                prompt += f"{seg1}{test_sentence}{seg2}"[:-1] + test_label_option
            return prompt

        example = single_prompt_func(orig_train_sentences[0], orig_train_labels[0])
        print(f"Sentence example: ||{example}||")

        params['prompt_func'] = prompt_func
        params['single_prompt_func'] = single_prompt_func

    elif params['dataset'][:9] == 'mit_movie':
        field_name = params['dataset'][10:]
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_slot_movies(field_name)
        """
        Actor 944
        Award 54
        Character_Name 225
        Director 415
        Genre 780
        Opinion 190
        Origin 178
        Plot 1459
        Quote 43
        Relationship 147
        Soundtrack 7
        Year 655
        """

        params['prompt_prefix'] = ""
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = f"{field_name}: "
        params['task_format'] = 'qa'
        params['num_tokens_to_predict'] = 1


        def prompt_func(params, train_sentences, train_labels, test_sentence, test_label_option=None):
            q_prefix = params["q_prefix"]
            a_prefix = params["a_prefix"]

            prompt = params['prompt_prefix']
            for x, y in zip(train_sentences, train_labels):
                prompt += f"{q_prefix}{x}\n{a_prefix}{y}"
                prompt += "\n\n"

            if test_label_option is None:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1]
            else:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1] + test_label_option
            return prompt

        params['prompt_func'] = prompt_func

    elif params['dataset'][:4] == 'atis':
        tag_name = params['dataset'][5:]
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_atis(tag_name)

        name2prefix = {
            "airline_name": "Airline name",
            "depart_time.period_of_day": "Depart time - Period of day",
            "depart_date.day_name": "Depart date - Day name"
        }

        params['prompt_prefix'] = ""
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = f"{name2prefix[tag_name]}: "
        params['task_format'] = 'qa'
        params['num_tokens_to_predict'] = 1

        def prompt_func(params, train_sentences, train_labels, test_sentence, test_label_option=None):
            q_prefix = params["q_prefix"]
            a_prefix = params["a_prefix"]

            prompt = params['prompt_prefix']
            for x, y in zip(train_sentences, train_labels):
                prompt += f"{q_prefix}{x}\n{a_prefix}{y}"
                prompt += "\n\n"

            if test_label_option is None:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1]
            else:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1] + test_label_option
            return prompt

        params['prompt_func'] = prompt_func

    else:
        raise NotImplementedError
    print('train set length: ', len(orig_train_sentences), ' test set length: ', len(orig_test_sentences), flush=True)
    return orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels
