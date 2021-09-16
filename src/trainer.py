"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import argparse
from collections import defaultdict
import copy
from datetime import datetime
import json
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import sys
import time
import torch

from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer
import wandb

from os.path import join, abspath, dirname

from src.data_utils.dataset import load_file, LAMADataset, load_json_file, load_files, load_relations
from src.data_utils.vocab import init_vocab
from src.p_tuning.modeling import PTuneForLAMA
from src.p_tuning.relation_classification import RelationClassifier


SUPPORT_MODELS = ['bert-base-cased', 'bert-large-cased',
                  'roberta-base', 'roberta-large']
                  


logger = logging.getLogger(__name__)

def set_seed(args):
    np.random.seed(args.model.seed)
    torch.manual_seed(args.model.seed)
    #if args.n_gpu > 0:
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.model.seed)



class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.device = args.device

        # save args
        if self.args.train:
            path = os.path.join(self.get_save_path("train_config"), "config.yaml")
        else:
            path = os.path.join(self.get_save_path(self.args.data.get("test").id), "config.yaml")
        OmegaConf.save(config=args, f=path)


        # load tokenizer
        tokenizer_src = 'roberta-large' if 'megatron' in self.args.model.name else self.args.model.name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)
        init_vocab(args)

        # check how many relations we need
        train_relation_ids = load_relations(self.args.data.train.relations_path)
        dev_relation_ids = load_relations(self.args.data.dev.relations_path)
        test_relation_ids = load_relations(self.args.data.test.relations_path)

        # load datasets and dataloaders
        self.train_data = load_files(self.args.data.train.pairs_path, map(lambda relation: f'{relation}/train.jsonl', train_relation_ids))
        self.dev_data = load_files(self.args.data.dev.pairs_path, map(lambda relation: f'{relation}/dev.jsonl', dev_relation_ids))
        self.test_data = load_files(self.args.data.test.pairs_path, map(lambda relation: f'{relation}/test.jsonl', test_relation_ids))

        if self.args.debug:
            self.train_data = self.train_data[:32]
            self.dev_data = self.train_data[:32]
            self.test_data = self.train_data[:32]
        if args.data.train.get("samples") is not None:
            self.train_data = np.random.choice(self.train_data, size=args.data.train.get("samples"), replace=False)
        if args.data.dev.get("samples") is not None:
            self.dev_data = np.random.choice(self.dev_data, size=args.data.dev.get("samples"), replace=False)
        if args.data.test.get("samples") is not None:
            self.test_data = np.random.choice(self.test_data, size=args.data.test.get("samples"), replace=False)
        
        # # I'm gonna do a dumb thing here where I just save the subsampled data here - this is only run once!
        # path = "/export/home/benjamin-newman-scratchpad/P-tuning/data/LAMA/fact-retrieval/original_subsampled"
        # for name, data in zip(("train.jsonl", "dev.jsonl", "test.jsonl"), (self.train_data, self.dev_data, self.test_data)):
        #     data_dict = defaultdict(list)
        #     for item in data:
        #         data_dict[item['predicate_id']].append(item)
        #     for relation in data_dict:
        #         os.makedirs(os.path.join(path, relation), exist_ok=True)
        #         with open(os.path.join(path, relation, name), "w") as f:
        #             f.write("\n".join([json.dumps(x) for x in data_dict[relation]]))



        self.test_set = LAMADataset('test', self.test_data, self.tokenizer, test_relation_ids, self.args)
        self.train_set = LAMADataset('train', self.train_data, self.tokenizer, train_relation_ids, self.args)
        self.dev_set = LAMADataset('dev', self.dev_data, self.tokenizer, dev_relation_ids, self.args)
        self.log(f"Train size: {len(self.train_set)}\nDev size: {len(self.dev_set)}\nTest size: {len(self.test_set)}")

        
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_set) if self.args.model.distributed else None
        self.train_loader = DataLoader(self.train_set, batch_size=self.args.model.batch_size, shuffle=(not self.args.model.distributed), drop_last=True, sampler=self.train_sampler)
        self.dev_loader = DataLoader(self.dev_set, batch_size=self.args.model.batch_size)
        self.test_loader = DataLoader(self.test_set, batch_size=self.args.model.batch_size)

        relclf_model = None
        if self.args.model.type == "moe":
            relclf_config = self.args.model.relclf
            with open_dict(relclf_config):
                relclf_config.train = args.train
                relclf_config.device = args.device
            relclf_model = RelationClassifier(self.args.model.relclf)
        print("relclf_model: ", relclf_model)
        self.model = PTuneForLAMA(args, self.device, relclf_model=relclf_model)
        if self.args.model.distributed:
            self.model = DDP(self.model)

        if not self.args.train and relclf_model is None:
            print("Loading saved models...")
            self.load()

        if self.args.train and self.args.model.type not in ("p-tuning", "ensemble"):
            wandb.init(project="p-tuning", config=OmegaConf.to_container(self.args, resolve=False))
            # wandb.watch(self.model)


    def evaluate(self, epoch_idx, evaluate_type):
        self.model.eval()
        if evaluate_type == 'test':
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
            split_metrics = defaultdict(list)
            for x_hs, x_ts, relations, templates in tqdm(loader, desc=f"{evaluate_type}-eval"):
                if evaluate_type == 'test' and not self.args.model.distributed:
                    loss, _hit1, metrics, top10 = self.model(x_hs, x_ts, relations, templates, return_candidates=True)
                    eval_output['x_hs'].extend(x_hs)
                    eval_output['x_ts'].extend(x_ts)
                    eval_output['relations'].extend(relations)
                    eval_output['templates'].extend(templates)
                    eval_output['predictions'].extend(top10)
                    eval_output['metrics'].extend(metrics)
                else:
                    outputs = self.model(x_hs, x_ts, relations, templates)
                    if self.args.model.distributed:
                        outputs = outputs.reshape(self.args.model.world_size, -1).mean(dim=0)
                    loss, _hit1, metrics = outputs
                hit1 += _hit1
                loss += loss.item()
                for sample in metrics:
                    for metric_name, value in sample.items():
                        split_metrics[metric_name].append(value)
            hit1 /= len(dataset)
            self.log("{} {} Epoch {} Loss: {} Hit@1:".format(self.args.data.get(evaluate_type).relations_id, evaluate_type, epoch_idx,
                                                          loss / len(dataset)), hit1)
            with open(os.path.join(self.get_save_path(self.args.data.get(evaluate_type).id), "results.json"), "w") as outfile:
                output_metrics = {"loss": loss.item() / len(dataset), "hit@1": hit1}
                for metric_name in split_metrics:
                    if "top10" in metric_name:
                        continue
                    output_metrics[metric_name] = np.mean(split_metrics[metric_name])
                if self.args.model.type == "ensemble":
                    p = torch.softmax(self.model.prompt_encoder.mixture_weights, dim=0)
                    output_metrics["ensemble_entropy"] = -(p * torch.log(p)).sum().item()
                    print(output_metrics["ensemble_entropy"])
                json.dump(output_metrics, outfile)

        if eval_output is not None:
            with open(os.path.join(self.get_save_path(self.args.data.get(evaluate_type).id), "predictions.json"), "w") as outfile:
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

    def get_save_path(self, sub_dir):
        #return join(self.args.out_dir, 'prompt_model', self.args.model_name, 'search', self.get_task_name(loading_ckpt),
        #            "_".join(self.args.relation_id))
        out_dir = f"{self.args.model.id}-{self.args.data.train.id}"
        if self.args.debug:
            out_dir += "-debug"
        # return to_absolute_path(join("out", out_dir, 'model'))
        out_dir = join(self.args.model.out_path_prefix, out_dir, sub_dir)
        os.makedirs(out_dir, exist_ok=True)
        return out_dir


    def get_checkpoint(self, epoch_idx, dev_hit1, test_hit1):
        ckpt_name = "epoch_{}_dev_{}_test_{}.ckpt".format(epoch_idx, round(dev_hit1 * 100, 4),
                                                          round(test_hit1 * 100, 4))
        return {'embedding': self.model.prompt_encoder.state_dict(), # embed_copy,
                'dev_hit@1': dev_hit1,
                'test_hit@1': test_hit1,
                'test_size': len(self.test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': self.args}

        
    def log(self, *message):
        """
        Prints message to stdout as well as the log file.
        """
        message = " ".join([str(item) for item in message])
        logger.info(message)

    def save(self, best_ckpt):
        ckpt_name = best_ckpt['ckpt_name']
        path = self.get_save_path("checkpoints")
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))
        self.log("# {} Checkpoint {} saved.".format(self.args.data.train.relations_id, ckpt_name))

    def load(self):
        if self.args.model.get("checkpoint_path") is None:
            path = self.get_save_path("checkpoints")
        else:
            path = self.args.model.checkpoint_path
        # choose the most recent checkpoint
        most_recent_ckpt = None
        for ckpt_name in os.listdir(path):
            # choose the most recent one I guess
            ckpt = torch.load(join(path, ckpt_name), map_location=self.device)
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
        if self.args.train:
            if self.args.model.distributed:
                if rank == 0:
                    print(dir(self.model))
                params = [{'params': self.model.module.prompt_encoder.parameters()}]
            else:
                params = [{'params': self.model.prompt_encoder.parameters()}]
                if self.args.model.finetune:
                    params.append({'params': self.model.model.parameters(), 'lr': 5e-6})
            optimizer = torch.optim.Adam(params, lr=self.args.model.lr, weight_decay=self.args.model.weight_decay)
            my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=self.args.model.decay_rate)

        max_epochs = 10 if self.args.debug else self.args.model.get("max_epochs", 100)
        grad_accum_steps = 128 // self.args.model.batch_size
        total_num_of_samples = 0
        for epoch_idx in range(max_epochs):
            start_time = time.time()
            self.log(f"Starting epoch {epoch_idx}")
            # check early stopping
            if epoch_idx > -1: # and (not self.args.distributed or rank == 0):
                if self.args.train:
                    dev_loss, dev_hit1 = self.evaluate(epoch_idx, 'dev')
                if epoch_idx == 0:
                    test_loss, test_hit1 = self.evaluate(epoch_idx, 'test')
                if epoch_idx > 0 and (dev_hit1 >= best_dev): # or self.args.only_evaluate: (this bit is for saving the checkpoint, but it's not necessary for our case)
                    test_loss, test_hit1 = self.evaluate(epoch_idx, 'test')
                    best_ckpt = self.get_checkpoint(epoch_idx, dev_hit1, test_hit1)
                    early_stop = 0
                    best_dev = dev_hit1
                    best_ckpt['opt'] = optimizer.state_dict()
                    self.save(best_ckpt)
                else:
                    early_stop += 1
                    if early_stop >= self.args.model.early_stop:
                        if not self.args.model.distributed or rank == 0:
                            self.save(best_ckpt)
                        self.log("{} Early stopping at epoch {}.".format(self.args.data.train.relations_id, epoch_idx))
                        return best_ckpt
            if not self.args.train:
                break
            self.log(f"Finish {epoch_idx} eval: {time.time() - start_time}")
            start_time = time.time()

            # run training
            if self.args.model.distributed:
                self.train_sampler.set_epoch(idx)

            hit1, num_of_samples = 0, 0
            tot_loss = 0
            for batch_idx, batch in tqdm(enumerate(self.train_loader)):
                self.model.train()
                loss, batch_hit1, metrics = self.model(*batch)
                # outputs = self.model(x_hs, x_ts, relations, templates)
                # if self.args.distributed:
                #     outputs = outputs.reshape(self.args.worldsize, -1).mean(dim=0)
                # loss, _hit1 = outputs
                hit1 += batch_hit1
                tot_loss += loss.item()
                num_of_samples += len(batch[0])
                total_num_of_samples += len(batch[0])


                loss.backward()
                if ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(self.train_loader)):
                    torch.cuda.empty_cache()
                    optimizer.step()
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()

                # merge metrics for logging
                merged_metrics = defaultdict(list)
                for sample in metrics:
                    for metric_name, value in sample.items():
                        merged_metrics[metric_name].append(value)
                for metric_name in merged_metrics:
                    if "top10" in metric_name:
                        continue
                    merged_metrics[metric_name] = np.mean(merged_metrics[metric_name])

                if batch_idx % 50 == 0:
                    if self.args.model.type not in ("p-tuning", "ensemble"):
                        wandb.log({"epoch": epoch_idx, "P@1": batch_hit1/len(batch), "loss": loss.item(), "batch_size": len(batch[0]),
                            **merged_metrics
                            }, step=total_num_of_samples)
            my_lr_scheduler.step()

            self.log(f"Finish {epoch_idx} train: {time.time() - start_time}")

        return best_ckpt


# Leaving this here as an example of how to get distributed working
# These methods should never run
# 
# def main_worker(gpu, world_size, args):
#     if args.distributed:
#         os.environ['MASTER_ADDR'] = 'localhost'
#         os.environ['MASTER_PORT'] = '29500'
#         dist.init_process_group("nccl", rank=gpu, world_size=world_size)
#         print(gpu)
#         args.rank = gpu
#         args.world_size = world_size
#     trainer = Trainer(args)
#     trainer.train(rank=gpu)
# 
# def main(relation_id=None):
#     args = construct_generation_args()
#     if relation_id:
#         args.relation_id = relation_id
#     if type(args.template) is not tuple:
#         args.template = eval(args.template)
#     assert type(args.template) is tuple
#     if args.relclf_do_train:
#         # shortcut for training relation classifier
#         init_vocab(args)
#         trainer = RelationClassifier(args)
#         trainer.train()
#     else:
#         if args.distributed:
#             mp.spawn(main_worker, nprocs=2, args=(2, args))
#         else:
#             main_worker(None, -1, args)
