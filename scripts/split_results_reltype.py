"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from collections import Counter
import hydra
from hydra.utils import to_absolute_path
import json
from omegaconf import DictConfig, OmegaConf, open_dict
import os
from transformers import AutoTokenizer

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

def get_save_path(cfg):
    out_dir = f"{cfg.model.id}-{cfg.data.train.id}"
    sub_dir = cfg.data.test.id
    if cfg.debug:
        out_dir += "-debug"
    # return to_absolute_path(join("out", out_dir, 'model'))
    return to_absolute_path(os.path.join(cfg.model.out_path_prefix, out_dir, sub_dir, "predictions.json"))

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # load the config
    cfg.device="cpu"
    resolve_paths(cfg)

    relations_file = to_absolute_path("data/LAMA/relations.jsonl")
    relations_info = {}
    with open(relations_file) as relf:
        for line in relf:
            relation_line = json.loads(line)
            # print(relation_line)
            relations_info[relation_line['relation']] = relation_line

    model = cfg.model.name
    predictions_file = get_save_path(cfg)
    print(predictions_file)
    with open(predictions_file) as predf:
        predictions = json.load(predf)

    tokenizer = AutoTokenizer.from_pretrained(model)
    if isinstance(predictions["predictions"][0], list):
        model_predictions = tokenizer.batch_decode([[p[0]] for p in predictions["predictions"]])
    else:
        model_predictions = tokenizer.batch_decode(predictions["predictions"])


    # split model prediction by type
    results = {}
    for rel_type in set([x['type'] for x in relations_info.values()]):
        results[rel_type] = []

    correct = Counter()
    total = Counter()
    for i in range(len(model_predictions)):
        if model_predictions[i] == predictions['x_ts'][i]:
            correct[relations_info[predictions['relations'][i]]['type']] += 1
        total[relations_info[predictions['relations'][i]]['type']] += 1

    for key in total:
        print(f"{key}: {correct[key]/total[key]:.3%}")




if __name__ == "__main__":
    main()
