"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from argparse import ArgumentParser
from omegaconf import OmegaConf
import os
import shlex
import subprocess
import sys



from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.utils import to_absolute_path
import torch

from trainer import Trainer, set_seed
from src.data_utils.dataset import load_relations
from src.p_tuning.relation_classification import RelationClassifier

def resolve_paths(conf):
    """Turns any paths in the config into the absolute path"""
    to_resolve = []
    for k in conf:
        if isinstance(conf[k], DictConfig):
            to_resolve.append(conf[k])
        elif "path" in k:
            conf[k] = to_absolute_path(conf[k])

    for sub_conf in to_resolve:
        resolve_paths(sub_conf)


def set_up_args(args):
    device_str = "cpu"
    if torch.cuda.is_available(): # and not args.no_cuda:
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None: # only should matter when training relclf because huggingface trainer doesn't let us specify our own device string
            device_str = "cuda:0"
        else:
            try:
                import GPUtil
                device_id = GPUtil.getFirstAvailable(order="load")[0]
                device_str = f"cuda:{device_id}"
                print(device_str)
            except ModuleNotFoundError:
                device_str = "cuda:0"
    args.device = device_str # torch.device(device_str)

    # replace all paths with their "absolute path" versions
    resolve_paths(args)
    set_seed(args)
    return args


@hydra.main(config_path="../configs", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    # print(cfg.data.data_id)

    cfg = set_up_args(cfg)

    if cfg.model.type == "p-tuning" or cfg.model.type == "ensemble":
        # train each relation separately
        train_relation_ids = load_relations(cfg.data.train.relations_path)
        for relation_id in train_relation_ids: # ["P937"]:
            # if relation_id < "P279":
            #     continue
            # update the training config
            print(f"{relation_id}")
            cfg.data.train.relations_path = relation_id
            cfg.data.train.relations_id = relation_id
            cfg.data.dev.relations_path = relation_id
            cfg.data.dev.relations_id = relation_id
            cfg.data.test.relations_path = relation_id
            cfg.data.test.relations_id = relation_id

            trainer = Trainer(cfg)
            trainer.train()  # rank=gpu)
            # import sys; sys.exit(0)
    elif cfg.model.type == "moe": # mixture of experts
        # we have to load the p-tuning config and the relclf configs
        # update the id to include the relclf
        relclf_model_cfg = OmegaConf.load(cfg.model.relclf.model_path)
        relclf_data_cfg = OmegaConf.load(cfg.model.relclf.data_path) # we need the training data to load in the correct model
        p_tuning_cfg = OmegaConf.load(cfg.model.ptuning.model_path)
        with open_dict(cfg):
            cfg.model = p_tuning_cfg
            cfg.model.relclf = OmegaConf.create({"model": OmegaConf.to_container(relclf_model_cfg, resolve=False), "data": OmegaConf.to_container(relclf_data_cfg, resolve=False)})
            # cfg.model.relclf.data = relclf_data_cfg

            # update some parameters
            cfg.model.ptuning_type = cfg.model.type
            cfg.model.ptuning_id = cfg.model.id
            cfg.model.ptuning_out_path_prefix = cfg.model.out_path_prefix

        cfg.model.id += f"-relclf-{cfg.model.relclf.model.id}-{cfg.model.relclf.data.train.id}"
        cfg.model.out_path_prefix = "out/moe"
        cfg.model.type = "moe"
        cfg = set_up_args(cfg)
        # print(OmegaConf.to_yaml(cfg, resolve=False))
        assert not cfg.train

        trainer = Trainer(cfg)
        trainer.train()
    elif cfg.model.type == "oracle": # mixture of experts
        # we have to load the p-tuning config and the relclf configs
        # update the id to include the relclf
        # relclf_model_cfg = OmegaConf.load(cfg.model.relclf.model_path)
        # relclf_data_cfg = OmegaConf.load(cfg.model.relclf.data_path) # we need the training data to load in the correct model
        p_tuning_cfg = OmegaConf.load(cfg.model.ptuning.model_path)
        with open_dict(cfg):
            cfg.model = p_tuning_cfg
            # cfg.model.relclf = OmegaConf.create({"model": OmegaConf.to_container(relclf_model_cfg, resolve=False), "data": OmegaConf.to_container(relclf_data_cfg, resolve=False)})
            # cfg.model.relclf.data = relclf_data_cfg

            # update some parameters
            cfg.model.ptuning_id = cfg.model.id
            cfg.model.ptuning_type = cfg.model.type
            cfg.model.ptuning_out_path_prefix = cfg.model.out_path_prefix

        cfg.model.id += f"_ptype-{cfg.model.ptuning_type}"
        cfg.model.out_path_prefix = "out/oracle"
        cfg.model.type = "oracle"
        cfg = set_up_args(cfg)
        # print(OmegaConf.to_yaml(cfg, resolve=False))
        assert not cfg.train

        trainer = Trainer(cfg)
        trainer.train()
    elif cfg.model.type == "relclf":
        trainer = RelationClassifier(cfg)
        trainer.train()
        # relclf should be evaluated here:
    else:
        trainer = Trainer(cfg)
        trainer.train() #rank=gpu)


if __name__ == "__main__":
    my_app()

