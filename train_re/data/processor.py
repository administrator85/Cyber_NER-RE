import csv
import pickle 
import os
import logging
from torch.utils.data import TensorDataset, Dataset
import torch
from collections import OrderedDict

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

import json

def tokenize(text, tokenizer):
    # berts tokenize ways
    # tokenize the [unused12345678910]
    D = [f"[unused{i}]" for i in range(10)]
    textraw = [text]
    for delimiter in D:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t)-1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in D:
            text += [t]
        else:
            tokens = tokenizer.tokenize(t, add_special_tokens=False)
            for tok in tokens:
                text += [tok]

    for idx, t in enumerate(text):
        if idx + 3 < len(text) and t == "[" and text[idx+1] == "[UNK]" and text[idx+2] == "]":
            text = text[:idx] + ["[MASK]"] + text[idx+3:]

    return text

class InputExampleWiki80(object):
    """A single training/test example for span pair classification."""

    def __init__(self, guid, sentence, span1, span2, ner1, ner2, label):
        self.guid = guid
        self.sentence = sentence
        self.span1 = span1
        self.span2 = span2
        self.ner1 = ner1
        self.ner2 = ner2
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, entity=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.entity = entity

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class wiki80Processor(DataProcessor):
    """Processor for the TACRED data set."""
    def __init__(self, data_path, use_prompt):
        super().__init__()
        self.data_dir = data_path

    @classmethod
    def _read_json(cls, input_file):
        data = []
        with open(input_file, "r", encoding='utf-8') as reader:
            all_lines = reader.readlines()
            for line in all_lines:
                ins = eval(line)
                data.append(ins)
        return data

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "val.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self, negative_label="no_relation"):
        data_dir = self.data_dir
        """See base class."""
        with open(os.path.join(data_dir,'rel2id.json'), "r", encoding='utf-8') as reader:
            re2id = json.load(reader)
        return re2id


    def _create_examples(self, dataset, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for example in dataset:
            sentence = example['token']
            examples.append(InputExampleWiki80(guid=None,
                            sentence=sentence,
                            # maybe some bugs here, I don't -1
                            span1=(example['h']['pos'][0], example['h']['pos'][1]),
                            span2=(example['t']['pos'][0], example['t']['pos'][1]),
                            ner1=None,
                            ner2=None,
                            label=example['relation']))
        return examples

def convert_examples_to_features_for_loss(examples, max_seq_length, tokenizer):
    print("#examples", len(examples))
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenize(example.text_a, tokenizer)
        tokens_b = tokenize(example.text_b, tokenizer)
        tokens_c = tokenize(example.text_c, tokenizer)

        # t_tokens = tokenize(example.entity, tokenizer)
        t_tokens = tokenizer(example.entity, add_special_tokens=False)["input_ids"]

        _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_seq_length - 4)
        tokens_b = tokens_b + ["[SEP]"] + tokens_c

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = example.label 

        len_t = len(t_tokens)
        normal_input_ids = input_ids[:]
        for idx, input_id in enumerate(input_ids):
            if idx + len_t < len(input_ids) and input_ids[idx:idx+len_t] == t_tokens:
                # [MASK] id = 103
                for j in range(len_t):
                    input_ids[j+idx] = 103

        # append 1 sample with 2 input
        features.append(
            [InputFeatures(
                input_ids=normal_input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                entity = t_tokens
            ),
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id,
                entity = t_tokens
            )]
        )
        
    print('#features', len(features))
    return features

def convert_examples_to_features(examples, max_seq_length, tokenizer, args, rel2id):
    """Loads a data file into a list of `InputBatch`s."""

    save_file = "./dataset/cached_wiki80.pkl"
    mode = "text"

    num_tokens = 0
    num_fit_examples = 0
    instances = []
    
    
    use_bert = "BertTokenizer" in tokenizer.__class__.__name__
    
    if False:
        with open(file=save_file, mode='rb') as fr:
            instances = pickle.load(fr)
        print('load preprocessed data from {}.'.format(save_file))

    else:
        print('loading..')
        for (ex_index, example) in enumerate(examples):
            

            """
                the relation between SUBJECT and OBJECT is .
                
            """

            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            tokens = []
            SUBJECT_START = "[subject_start]"
            SUBJECT_END = "[subject_end]"
            OBJECT_START = "[object_start]"
            OBJECT_END = "[object_end]"


            if mode.startswith("text"):
                for i, token in enumerate(example.sentence):
                    if i == example.span1[0]:
                        tokens.append(SUBJECT_START)
                    if i == example.span2[0]:
                        tokens.append(OBJECT_START)
                    # for sub_token in tokenizer.tokenize(token):
                    #     tokens.append(sub_token)
                    if i == example.span1[1]:
                        tokens.append(SUBJECT_END)
                    if i == example.span2[1]:
                        tokens.append(OBJECT_END)

                    tokens.append(token)

            SUBJECT = " ".join(example.sentence[example.span1[0]: example.span1[1]])
            OBJECT = " ".join(example.sentence[example.span2[0]: example.span2[1]])
            SUBJECT_ids = tokenizer(" "+SUBJECT, add_special_tokens=False)['input_ids']
            OBJECT_ids = tokenizer(" "+OBJECT, add_special_tokens=False)['input_ids']
            
    
            # add prompt [T_n] and entity marker [obj] to enrich the context.

            if args.use_template_words:
                prompt = f"[sub] {SUBJECT} [sub] {tokenizer.mask_token} [obj] {OBJECT} [obj] ."
            else:
                prompt = f"{SUBJECT} {tokenizer.mask_token} {OBJECT}."
            
            if ex_index == 0:
                input_text = " ".join(tokens)
                logger.info(f"input text : {input_text}")
                logger.info(f"prompt : {prompt}")
                logger.info(f"label : {example.label}")
            inputs = tokenizer(
                prompt,
                " ".join(tokens),
                truncation="longest_first",
                max_length=max_seq_length,
                padding="max_length",
                add_special_tokens=True
            )
         
            
            # find the subject and object tokens, choose the first ones
            sub_st = sub_ed = obj_st = obj_ed = -1
            for i in range(len(inputs['input_ids'])):
                if sub_st == -1 and inputs['input_ids'][i:i+len(SUBJECT_ids)] == SUBJECT_ids:
                    sub_st = i
                    sub_ed = i + len(SUBJECT_ids)
                if obj_st == -1 and inputs['input_ids'][i:i+len(OBJECT_ids)] == OBJECT_ids:
                    obj_st = i
                    obj_ed = i + len(OBJECT_ids)
            
            assert sub_st != -1 and obj_st != -1


            num_tokens += sum(inputs['attention_mask'])


            if sum(inputs['attention_mask']) > max_seq_length:
                pass
                # tokens = tokens[:max_seq_length]
            else:
                num_fit_examples += 1

            x = OrderedDict()
            x['input_ids'] = inputs['input_ids']
            if use_bert: x['token_type_ids'] = inputs['token_type_ids']
            x['attention_mask'] = inputs['attention_mask']
            x['label'] = rel2id[example.label]
            x['so'] =[sub_st, sub_ed, obj_st, obj_ed]

            instances.append(x)


        with open(file=save_file, mode='wb') as fw:
            pickle.dump(instances, fw)
        print('Finish save preprocessed data to {}.'.format( save_file))

    input_ids = [o['input_ids'] for o in instances]
    attention_mask = [o['attention_mask'] for o in instances]
    if use_bert: token_type_ids = [o['token_type_ids'] for o in instances]
    labels = [o['label'] for o in instances]
    so = torch.tensor([o['so'] for o in instances])


    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    if use_bert: token_type_ids = torch.tensor(token_type_ids)
    labels = torch.tensor(labels)

    logger.info("Average #tokens: %.2f" % (num_tokens * 1.0 / len(examples)))
    logger.info("%d (%.2f %%) examples can fit max_seq_length = %d" % (num_fit_examples,
                num_fit_examples * 100.0 / len(examples), max_seq_length))

    if use_bert:
        dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels, so)
    else:
        dataset = TensorDataset(input_ids, attention_mask, labels, so)
    
    return dataset

def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()            

def get_dataset(mode, args, tokenizer, processor):

    if mode == "train":
        examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        examples = processor.get_dev_examples(args.data_dir)
    elif mode == "test":
        examples = processor.get_test_examples(args.data_dir)
    else:
        raise Exception("mode must be in choice [trian, dev, test]")
    
        # normal relation extraction task
    dataset = convert_examples_to_features(
        examples, args.max_seq_length, tokenizer, args, processor.get_labels()
    )
    return dataset

def collate_fn(batch):
    pass

processors = { "wiki80": wiki80Processor}