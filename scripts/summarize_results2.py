from argparse import ArgumentParser
from collections import defaultdict
import os

import torch
import numpy as np

def load(path):
    most_recent_ckpt = None
    for ckpt_name in os.listdir(path):
        # choose the most recent one I guess
        ckpt = torch.load(os.path.join(path, ckpt_name), map_location=torch.device('cpu'))
        most_recent_ckpt = ckpt if (
                most_recent_ckpt is None or ckpt['time'] > most_recent_ckpt['time']
        ) else most_recent_ckpt
    return most_recent_ckpt

def main(args):
    accuracies = defaultdict(list)
    for relation in os.listdir(args.eval_output_dir):
        ckpt = load(os.path.join(args.eval_output_dir, relation))
        accuracies['dev'].append(ckpt["dev_hit@1"])
        accuracies['test'].append(ckpt["test_hit@1"])

    for split in accuracies:
        print(split, np.mean(accuracies[split]))


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("eval_output_dir", type=str)
    args = argparser.parse_args()
    main(args)

