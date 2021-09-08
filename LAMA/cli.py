from collections import defaultdict
import copy
import json
import os
import time
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer

from os.path import join, abspath, dirname

from LAMA.data_utils.dataset import load_file, LAMADataset, load_json_file, load_files, load_relations
from LAMA.data_utils.vocab import init_vocab
from LAMA.p_tuning.modeling import PTuneForLAMA
from LAMA.p_tuning.relation_classification import RelationClassifier


SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased',
                  'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',
                  'roberta-base', 'roberta-large',
                  'megatron_11b']


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--relation_id", type=str, default="P1001", help="relation, comma-separated list of relations,\
                                                                         or path to file with one relation per line")
    parser.add_argument("--model_name", type=str, default='megatron_11b', choices=SUPPORT_MODELS)
    parser.add_argument("--pseudo_token", type=str, default='[PROMPT]')

    parser.add_argument("--t5_shard", type=int, default=0)
    parser.add_argument("--mid", type=int, default=0)
    parser.add_argument("--template", type=str, default="(3, 3, 3)")
    parser.add_argument("--early_stop", type=int, default=20)

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--decay_rate", type=float, default=0.98)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--batch_size", type=int, default=8)

    # distributed training
    parser.add_argument("--distributed", type=bool, default=False)
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--rank", type=int, default=-1)


    # lama configuration
    parser.add_argument("--only_evaluate", type=bool, default=False)
    parser.add_argument("--use_original_template", type=bool, default=False)
    parser.add_argument("--use_lm_finetune", type=bool, default=False)
    parser.add_argument("--ood_test_relations", type=str, default=None, help="relation, comma-spearated list of relations,\
                                                                             or path to file with one relation per line\
                                                                             whose data we didn't train on and we want\
                                                                             to use for ood testing")
    parser.add_argument("--relation_templates", type=str, default="relations.jsonl", help="Path from data_dir to the map from relations to templates")

    # adaptive prompt
    parser.add_argument("--use_adaptive_prompt", type=bool, default=False, help="Do we use a prompt that depends on the user input?")
    # parser.add_argument("--use_adaptive_prompt_train", type=bool, default=False, help="Do we use a prompt that depends on the user input?")
    parser.add_argument("--prompt_encoder_name", type=str, default="identity", help="Which prompt encoder to use")
    parser.add_argument("--debug", action="store_true")

    # relation classifier
    parser.add_argument("--relclf_model_name", type=str, default=None, help="Huggingface Transformers name or path to fine-tuned model")
    parser.add_argument("--relclf_do_train", type=bool, default=False)
    parser.add_argument("--relclf_prompt_path", type=str, default="out/LAMA/prompt_model/bert-base-cased/search/bert-base-cased_shared_template_(3, 3, 3)_fixed_seed_34_None")
    parser.add_argument("--relclf_use_gold_relations", type=bool, default=False)

    parser.add_argument("--vocab_strategy", type=str, default="shared", choices=['original', 'shared', 'lama'])
    parser.add_argument("--lstm_dropout", type=float, default=0.0)

    # directories
    parser.add_argument("--data_dir", type=str, default=join(abspath(dirname(__file__)), '../data/LAMA/fact-retrieval/original/'))
    parser.add_argument("--out_dir", type=str, default=join(abspath(dirname(__file__)), '../out/LAMA'))

    # MegatronLM 11B
    parser.add_argument("--checkpoint_dir", type=str, default=None) # default=join(abspath(dirname(__file__)), '../checkpoints'))

    # Misc
    parser.add_argument("--notes", type=str, default=None)

    args = parser.parse_args()

    # post-parsing args

    device_str = "cpu"
    if torch.cuda.is_available() and not args.no_cuda:
        try:
            import GPUtil
            device_id = GPUtil.getFirstAvailable(order="load")[0]
            device_str = f"cuda:{device_id}"
            print(device_str)
        except ModuleNotFoundError:
            device_str = "cuda:1"

    args.device = torch.device(device_str)
    # args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.template = eval(args.template) if type(args.template) is not tuple else args.template

    assert type(args.template) is tuple

    set_seed(args)

    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device
        # self.device = args.device # 'cuda:0' if self.args.model_name != 't5-11b' else 'cuda:{}'.format(self.args.t5_shard * 4)

        if self.args.use_original_template and (not self.args.use_lm_finetune) and (not self.args.only_evaluate) and (not self.args.use_adaptive_prompt):
            raise RuntimeError("""If use args.use_original_template is True, 
            either args.use_lm_finetune or args.only_evaluate should be True.""")

        # initialize logging
        self._log_filename = self.init_log()
        self.log("#"*20 + "\n", self.args)

        # load tokenizer
        tokenizer_src = 'roberta-large' if 'megatron' in self.args.model_name else self.args.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
        init_vocab(args)

        # check how many relations we need
        args.relation_id = load_relations(self.args.relation_id)
        print(args.relation_id)

        # load datasets and dataloaders
        # self.relation, self.data_path_pre, self.data_path_post = self.get_TREx_parameters()

        data_prefix = self.args.data_dir # join(self.args.data_dir, "fact-retrieval/original/")
        self.train_data = load_files(data_prefix, map(lambda relation: f'{relation}/train.jsonl', self.args.relation_id))
        self.dev_data = load_files(data_prefix, map(lambda relation: f'{relation}/dev.jsonl', self.args.relation_id))
        if self.args.ood_test_relations is not None:
            self.args.ood_test_relations = load_relations(self.args.ood_test_relations)
            self.test_data = load_files(data_prefix, map(lambda relation: f'{relation}/test.jsonl', self.args.ood_test_relations))
            print("ood:", self.args.ood_test_relations)
        else:
            self.test_data = load_files(data_prefix, map(lambda relation: f'{relation}/test.jsonl', self.args.relation_id))
        # self.test_data = load_files("data/LANKA/bert_data/wiki_uni", self.args.relation_id)
        # self.train_data = load_file(join(self.args.data_dir, self.data_path_pre + 'train' + self.data_path_post))
        # self.dev_data = load_file(join(self.args.data_dir, self.data_path_pre + 'dev' + self.data_path_post))
        # self.test_data = load_file(join(self.args.data_dir, self.data_path_pre + 'test' + self.data_path_post))
        # self.test_data = load_json_file("data/LANKA/bert_data/wiki_uni/" + self.args.relation_id)

        if self.args.debug:
            self.train_data = self.train_data[:32]

        # self.dev_data = self.dev_data[:512]
        # self.test_data = self.dev_data[:512]
        self.test_set = LAMADataset('test', self.test_data, self.tokenizer, self.args)
        self.train_set = LAMADataset('train', self.train_data, self.tokenizer, self.args)
        self.dev_set = LAMADataset('dev', self.dev_data, self.tokenizer, self.args)
        self.log(f"Train size: {len(self.train_set)}\nDev size: {len(self.dev_set)}\nTest size: {len(self.test_set)}")
        os.makedirs(self.get_save_path(), exist_ok=True)

        
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set) if self.args.distributed else None
        self.train_loader = DataLoader(self.train_set, batch_size=self.args.batch_size, shuffle=(not self.args.distributed), drop_last=True, sampler=self.train_sampler)
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.args.batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.batch_size)

        relclf_model = RelationClassifier(self.args) if self.args.relclf_model_name is not None else None
        self.model = PTuneForLAMA(args, self.device, self.args.template, relclf_model=relclf_model)
        if self.args.distributed:
            self.model = DDP(self.model)

        if self.args.only_evaluate and relclf_model is None:
            self.load()

    # def get_TREx_parameters(self):
    #     relation = load_file(join(self.args.data_dir, "single_relations/{}.jsonl".format(self.args.relation_id)))[0]
    #     data_path_pre = "fact-retrieval/original/{}/".format(self.args.relation_id)
    #     data_path_post = ".jsonl"
    #     return relation, data_path_pre, data_path_post

    def evaluate(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'Test':
            loader = self.test_loader
            dataset = self.test_set
            eval_output = defaultdict(list)
        else:
            loader = self.dev_loader
            dataset = self.dev_set
            eval_output = None
        with torch.no_grad():
            self.model.eval()
            hit1, loss = 0, 0
            for x_hs, x_ts, relations, templates in loader:
                if False and self.args.extend_data:
                    loss, _hit1 = self.model.test_extend_data(x_hs, x_ts, relations, templates)
                elif evaluate_type == 'Test' and not self.args.distributed:
                    loss, _hit1, top10 = self.model(x_hs, x_ts, relations, templates, return_candidates=True)
                    eval_output['x_hs'].extend(x_hs)
                    eval_output['x_ts'].extend(x_ts)
                    eval_output['relations'].extend(relations)
                    eval_output['templates'].extend(templates)
                    eval_output['predictions'].extend(top10)
                else:
                    # loss, _hit1 = self.model(x_hs, x_ts, relations, templates)
                    outputs = self.model(x_hs, x_ts, relations, templates)
                    if self.args.distributed:
                        outputs = outputs.reshape(self.args.world_size, -1).mean(dim=0)
                    loss, _hit1 = outputs
                hit1 += _hit1
                loss += loss.item()
            hit1 /= len(dataset)
            self.log("{} {} Epoch {} Loss: {} Hit@1:".format(self.args.relation_id[0], evaluate_type, epoch_idx,
                                                          loss / len(dataset)), hit1)
            if eval_output is not None:
                with open(os.path.join(self.log_path, "predictions.json"), "w") as outfile:
                    json.dump(eval_output, outfile)
        return loss, hit1

    def get_task_name(self, loading_ckpt=False):
        if self.args.only_evaluate and not loading_ckpt:
            return "_".join([self.args.model_name + ('_' + self.args.vocab_strategy), 'only_evaluate'])
        names = [self.args.model_name + ('_' + self.args.vocab_strategy),
                 "template_{}".format(self.args.template if not self.args.use_original_template else 'original'),
                 "fixed" if not self.args.use_lm_finetune else "fine-tuned",
                 "seed_{}".format(self.args.seed),
                 self.args.prompt_encoder_name if self.args.use_adaptive_prompt else "None"]
        if self.args.notes is not None:
            names.append(f"note-{self.args.notes}")
        return "_".join(names)

    def get_save_path(self, loading_ckpt=False):
        #return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'search', self.get_task_name(loading_ckpt),
        #            "_".join(self.args.relation_id))
        return join(self.args.out_dir, 'model')

    def get_log_path(self):
        return join(self.args.out_dir, 'logs')
        # if os.path.isfile(self.args.relation_id):
        #     relation_id = os.path.splitext(os.path.basename(self.args.relation_id))[0]
        # else:
        #     relation_id = self.args.relation_id.replace(",", "_")
        #
        # return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'logs', self.get_task_name(loading_ckpt=True),
        #             relation_id)

    def get_checkpoint(self, epoch_idx, dev_hit1, test_hit1):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, round(dev_hit1 * 100, 4),
                                                          round(test_hit1 * 100, 4))
        return {'embedding': copy.deepcopy(self.model.prompt_encoder.state_dict()),
                'dev_hit@1': dev_hit1,
                'test_hit@1': test_hit1,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

    def init_log(self):
        self.log_path = self.get_log_path()
        os.makedirs(self.log_path, exist_ok=True)
        log_basename = "train.log"
        if self.args.only_evaluate and self.args.ood_test_relations is not None:
            log_basename = "ood_eval.log"
        elif self.args.only_evaluate: # this one shouldn't be used too much...
            log_basename = "eval.log"
        return os.path.join(self.log_path, log_basename)
        
    def log(self, *message):
        """
        Prints message to stdout as well as the log file.
        """
        print(*message)
        message = " ".join([str(item) for item in message])
        with open(self._log_filename, "a") as f:
            f.write(f"{message}\n")

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path()
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        # print("# Prompt:", self.model.prompt)
        self.log("# {} Checkpoint {} saved.".format(self.args.relation_id, ckpt_name))

    def load(self):
        # path = self.get_save_path(loading_ckpt=True)
        if self.args.checkpoint_dir is None:
            path = self.get_save_path(loading_ckpt=True)
        else:
            path = self.args.checkpoint_dir
        # choose the most recent checkpoint
        most_recent_ckpt = None
        for ckpt_name in os.listdir(path):
            # choose the most recent one I guess
            ckpt = torch.load(join(path, ckpt_name))
            most_recent_ckpt = ckpt if (
                    most_recent_ckpt is None or ckpt['time'] > most_recent_ckpt['time']
            ) else most_recent_ckpt
        if most_recent_ckpt is not None:
            self.model.prompt_encoder.load_state_dict(most_recent_ckpt['embedding'])
        else:
            self.log(f"Unable to load a prompt encoder checkpoint from {path}. This is OK if you're not using an optimized prompt.")

    def train(self, rank=-1):
        best_dev, early_stop, has_adjusted = 0, 0, True
        best_ckpt = None
        if not self.args.only_evaluate:
            if self.args.distributed:
                if rank == 0:
                    print(dir(self.model))
                params = [{'params': self.model.module.prompt_encoder.parameters()}]
            else:
                params = [{'params': self.model.prompt_encoder.parameters()}]
                if self.args.use_lm_finetune:
                    params.append({'params': self.model.model.parameters(), 'lr': 5e-6})
            optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)

        max_epochs = 10 if self.args.debug else 100
        for epoch_idx in range(max_epochs):
            start_time = time.time()
            self.log(f"Starting epoch {epoch_idx}")
            # check early stopping
            if epoch_idx > -1: # and (not self.args.distributed or rank == 0):
                dev_loss, dev_hit1 = self.evaluate(epoch_idx, 'Dev')
                if epoch_idx == 0:
                    test_loss, test_hit1 = self.evaluate(epoch_idx, 'Test')
                if epoch_idx > 0 and (dev_hit1 >= best_dev): # or self.args.only_evaluate: (this bit is for saving the checkpoint, but it's not necessary for our case)
                    test_loss, test_hit1 = self.evaluate(epoch_idx, 'Test')
                    best_ckpt = self.get_checkpoint(epoch_idx, dev_hit1, test_hit1)
                    early_stop = 0
                    best_dev = dev_hit1
                else:
                    early_stop += 1
                    if early_stop >= self.args.early_stop:
                        if not self.args.distributed or rank == 0:
                            self.save(best_ckpt)
                        self.log("{} Early stopping at epoch {}.".format(self.args.relation_id, epoch_idx))
                        return best_ckpt
            if self.args.only_evaluate:
                break
            self.log(f"Finish {epoch_idx} eval: {time.time() - start_time}")
            start_time = time.time()

            # run training
            if self.args.distributed:
                self.train_sampler.set_epoch(idx)
            hit1, num_of_samples = 0, 0
            tot_loss = 0
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                loss, batch_hit1 = self.model(*batch)
                # outputs = self.model(x_hs, x_ts, relations, templates)
                # if self.args.distributed:
                #     outputs = outputs.reshape(self.args.worldsize, -1).mean(dim=0)
                # loss, _hit1 = outputs
                hit1 += batch_hit1
                tot_loss += loss.item()
                num_of_samples += len(batch[0])

                loss.backward()
                torch.cuda.empty_cache()
                optimizer.step()
                torch.cuda.empty_cache()
                optimizer.zero_grad()
            my_lr_scheduler.step()

            self.log(f"Finish {epoch_idx} train: {time.time() - start_time}")
        if not self.args.only_evaluate:
            self.save(best_ckpt)

        return best_ckpt

    # def train_distributed(self):
    #     def run(rank, world_size):
    #         dist.init_process_group("gloo", rank=rank, worlds_size=world_size)
    #         ddp_model = DDP(self.model, device_ids=[rank])
    #         params = [{'params': ddp_model.model.prompt_encoder.parameters()}]
    #         if self.args.use_lm_finetune:
    #             params.append({'params': ddp_model.model.model.parameters(), 'lr': 5e-6})
    #         optimizer = torch.optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
    #         my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.decay_rate)
    #         if epoch_idx > -1 and torch.distributed.get_rank() == 0: # only run eval on one node
    #             dev_loss, dev_hit1 = self.evaluate(epoch_idx, 'Dev')
    #             if epoch_idx == 0:
    #                 test_loss, test_hit1 = self.evaluate(epoch_idx, 'Test')
    #             if epoch_idx > 0 and (dev_hit1 >= best_dev):
    #                 test_loss, test_hit1 = self.evaluate(epoch_idx, 'Test')
    #                 best_ckpt = self.get_checkpoint(epoch_idx, dev_hit1, test_hit1)
    #                 early_stop = 0
    #                 best_dev = dev_hit1
    #             else:
    #                 early_stop += 1
    #                 if early_stop >= self.args.early_stop:
    #                     self.save(best_ckpt)
    #                     self.log("{} Early stopping at epoch {}.".format(self.args.relation_id, epoch_idx))
    #                     return best_ckpt


def main_worker(gpu, world_size, args):
    if args.distributed:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group("nccl", rank=gpu, world_size=world_size)
        print(gpu)
        args.rank = gpu
        args.world_size = world_size
    trainer = Trainer(args)
    trainer.train(rank=gpu)

def main(relation_id=None):
    args = construct_generation_args()
    if relation_id:
        args.relation_id = relation_id
    if type(args.template) is not tuple:
        args.template = eval(args.template)
    assert type(args.template) is tuple
    if args.relclf_do_train:
        # shortcut for training relation classifier for now
        init_vocab(args)
        trainer = RelationClassifier(args)
        trainer.train()
        print(trainer.predict(["Ottawa works in the field of [MASK]."]))
    else:
        if args.distributed:
            mp.spawn(main_worker, nprocs=2, args=(2, args))
        else:
            main_worker(None, -1, args)


if __name__ == '__main__':
    main()
