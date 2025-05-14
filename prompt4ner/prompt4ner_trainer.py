import argparse
import math
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig

from prompt4ner import models
from prompt4ner import sampling
from prompt4ner import util
from prompt4ner.entities import Dataset
from prompt4ner.evaluator import Evaluator
from prompt4ner.input_reader import JsonInputReader, BaseInputReader
from tqdm import tqdm


SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

class Prompt4NERTrainer():
    """ Joint entity and relation extraction training and evaluation """

    def __init__(self,model_path):
        super().__init__()


        self.args = argparse.Namespace()
        # args  read from json fil
        self.args.boundary_threshold = 0.0
        self.args.cache_path = None
        self.args.cls_threshold = 0.0
        self.args.config = "configs/eval.conf"
        self.args.cpu = False
        self.args.dataset_path = "/data/ner/datasets/ner/valid.json"
        self.args.debug = False
        self.args.decoder_layers = 3
        self.args.device_id = 1
        self.args.epochs = 50
        self.args.eval_batch_size = 1
        self.args.example_count = None
        self.args.freeze_transformer = False
        self.args.label = "security_eval"
        self.args.last_layer_for_loss = 1
        self.args.local_rank = -1
        self.args.log_path = "/data/ner/PromptNER_origin/data/few-shot"
        self.args.loss_boundary_weight = 2.0
        self.args.loss_class_weight = 2.0
        self.args.lowercase = False
        self.args.lstm_layers = 3
        self.args.match_boundary_weight = 2.0
        self.args.match_class_weight = 2.0
        self.args.match_solver = "hungarian"
        self.args.model_path =model_path
        self.args.model_type = "prompt4ner"
        self.args.nil_weight = -1.0
        self.args.no_duplicate = True
        self.args.no_overlapping = False
        self.args.no_partial_overlapping = True
        self.args.pool_type = "max"
        self.args.prompt_individual_attention = False
        self.args.prompt_length = 2
        self.args.prompt_number = 50
        self.args.prompt_type = "soft"
        self.args.prop_drop = 0.5
        self.args.repeat_gt_entities = 45
        self.args.sampling_processes = 0
        self.args.seed = 47
        self.args.sentence_individual_attention = True
        self.args.split_epoch = 5
        self.args.store_examples = False
        self.args.store_predictions = True
        self.args.tokenizer_path = model_path
        self.args.types_path = "/data/ner/datasets/ner/ner_types.json"
        self.args.use_masked_lm = False
        self.args.weight_decay = 0.01
        self.args.withimage = False
        self.args.world_size = -1

        self._device = torch.device('cuda', 1)

        self.model_path = model_path
        # byte-pair encoding

        self._tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                        local_files_only = True,
                                                        do_lower_case=False,
                                                        use_fast = False)

        self._processor = None
        
        # path to export predictions to
        self._predictions_path = os.path.join('./', 'predictions_%s_epoch_%s.json')

        # path to export relation extraction examples to
        self._examples_path = os.path.join('./', 'examples_%s_%s_epoch_%s.html')

        self.result = None  
    
    def load_model(self, input_reader):
        print(f"*****args.tokenizer_path: {self.args.tokenizer_path}; args.model_path: {self.args.model_path}")
        args = self.args
        # create model
        model_class = models.get_model(args.model_type)
        
        # load model
        config = AutoConfig.from_pretrained(self.model_path, cache_dir=args.cache_path)
        model = model_class.from_pretrained(self.model_path,
                                            ignore_mismatched_sizes=True,
                                            local_files_only = True,
                                            config = config,
                                            # Prompt4NER model parameters
                                            entity_type_count=input_reader.entity_type_count,
                                            prop_drop=args.prop_drop,
                                            freeze_transformer=args.freeze_transformer,
                                            lstm_layers = args.lstm_layers,
                                            decoder_layers = args.decoder_layers,
                                            pool_type = args.pool_type,
                                            prompt_individual_attention = args.prompt_individual_attention,
                                            sentence_individual_attention = args.sentence_individual_attention,
                                            use_masked_lm = args.use_masked_lm,
                                            last_layer_for_loss = args.last_layer_for_loss,
                                            split_epoch = args.split_epoch,
                                            clip_v = None,
                                            prompt_length = args.prompt_length,
                                            prompt_number = args.prompt_number,
                                            prompt_token_ids = input_reader.prompt_token_ids)
        return model

    def eval(self, dataset_path: str, types_path: str, input_reader_cls: BaseInputReader):

        dataset_label = 'test'

        # read datasets
        input_reader = input_reader_cls(
            types_path, 
            self._tokenizer, 
            self._processor,
            None, 
            random_mask_word = False ,   
            repeat_gt_entities = 45,      
            prompt_length = 2,
            prompt_type = "soft",
            prompt_number = 50)
        
        input_reader.read({dataset_label: dataset_path})

        model = self.load_model(input_reader)
        
        # 切换到评估模式
        model.eval()
        
        # evaluate
        self._eval(model, input_reader.get_dataset(dataset_label), input_reader)


    def _eval(self, model: torch.nn.Module, dataset, input_reader: JsonInputReader, epoch: int = 0):
        args = self.args

        # create evaluator
        evaluator = Evaluator(dataset, input_reader, self._tokenizer, args.no_overlapping, args.no_partial_overlapping, args.no_duplicate, self._predictions_path,
                              self._examples_path, args.example_count, epoch, dataset.label, cls_threshold = args.cls_threshold, boundary_threshold = args.boundary_threshold, save_prediction = args.store_predictions)

        # create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)

        world_size = 1
        eval_sampler = None

        if isinstance(dataset, Dataset):
            data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=args.sampling_processes, collate_fn=sampling.collate_fn_padding, sampler=eval_sampler)
        else:
            data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, drop_last=False, collate_fn=sampling.collate_fn_padding, sampler=eval_sampler)

        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.document_count / (args.eval_batch_size * world_size))
            for batch in tqdm(data_loader, total=total, desc='Evaluate epoch %s' % epoch):
                # move batch to selected device
                batch = util.to_device(batch, self._device)

                # run model (forward pass)
                entity_logits, p_left, p_right, _, outputs = model(
                    encodings=batch['encodings'], 
                    context_masks=batch['context_masks'], 
                    raw_context_masks=batch['raw_context_masks'], 
                    inx4locator = batch["inx4locator"],
                    pos_encoding = batch["pos_encoding"],
                    seg_encoding = batch['seg_encoding'], 
                    context2token_masks=batch['context2token_masks'], 
                    token_masks=batch['token_masks'],
                    image_inputs = batch['image_inputs'], 
                    meta_doc = batch['meta_doc'], 
                    evaluate = True)

                # evaluate batch
                evaluator.eval_batch(entity_logits, p_left, p_right, outputs, batch)
        # print(f"*****predictions: {evaluator._raw_preds}")
        # added by wyf
        # entities 去重
        for doc in evaluator._raw_preds:
            seen = set()
            unique_entities = []
            for entity in doc["entities"]:
                # 使用 start+end+type 作为唯一标识
                identifier = (entity["start"], entity["end"], entity["entity_type"])
                if identifier not in seen:
                    seen.add(identifier)
                    unique_entities.append(entity)
            doc["entities"] = unique_entities
        
        self.result = evaluator._raw_preds
        
        import json
        json.dump(evaluator._raw_preds, open("/data/ner/test/predictions.json", "w"))

        return "evaluated"
