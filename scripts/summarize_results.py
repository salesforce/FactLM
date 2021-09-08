from argparse import ArgumentParser
from collections import defaultdict
import json
import re

import numpy as np

# sample line:
# P17 Dev Epoch 0 Loss: 0.023765938356518745 Hit@1: 0.20512820512820512
re_template = "(P\d+) (Dev|Test) Epoch (\d+) Loss: (\d+\.\d+e?-?\d+) Hit@1: (\d+\.\d+)"

def main(args):
    results_dict = defaultdict(dict)
    with open(args.eval_output_filename) as f:
        for line in f:
            line = line.strip()
            match = re.match(re_template, line)
            if match is not None:
                relation, split, epoch, loss, hit_at_1 = match.groups()
                results_dict[split][relation] = {"epoch": int(epoch), "loss": float(loss), "hit@1": float(hit_at_1)}
    # summarize
    relations_info = {}
    with open("data/LAMA/relations.jsonl") as relations_f:
        relations_info_list = [json.loads(line) for line in relations_f]
        for line in relations_info_list:
            relations_info[line["relation"]] = line

    # print(relations_info)
    for split in results_dict:
        accuracies_subset = defaultdict(list)
        result_info = results_dict[split]
        for relation in result_info:
            accuracies_subset[relations_info[relation]["type"]].append(result_info[relation]["hit@1"])
            accuracies_subset["all"].append(result_info[relation]["hit@1"])


        # accuracies = [result["hit@1"] for result in results_dict[split].values()]
        # print(accuracies)
        print(f"-----{split}-----")
        for subsplit_name, accs in accuracies_subset.items():
            print(f"{subsplit_name}: {np.mean(accs)} (num relations: {len(accs)})")
        # print(f"{split}: {np.mean(accuracies)} (out of {len(accuracies)})")
    # print(results_dict)



if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("eval_output_filename", type=str)
    args = argparser.parse_args()
    main(args)
