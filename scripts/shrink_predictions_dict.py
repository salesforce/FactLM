import json
import os
import shutil
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizers = {
        "bert-base-cased": AutoTokenizer.from_pretrained("bert-base-cased"),
        "bert-large-cased": AutoTokenizer.from_pretrained("bert-base-cased"),
        "roberta-large": AutoTokenizer.from_pretrained("bert-base-cased"),
}

def main():
    with open("all_results_with_predictions.json") as f:
        data = json.load(f)

    path_prefix = "/export/home/benjamin-newman-scratchpad/P-tuning/"
    path_prefix_len = len(path_prefix)
    for i, path in tqdm(enumerate(data['path']), total=len(data['path'])):
        # if i < 357:
        #     continue
        # print(i, path)
        assert path[:path_prefix_len] == path_prefix

        new_out_dir = f"smaller_preds/{path[path_prefix_len:]}"
        try:
            # os.makedirs(new_out_dir, exist_ok=True)
            os.makedirs(new_out_dir)
            # os.makedirs(new_out_dir)
        except FileExistsError:
            # print("\texists, skipping....")
            continue

        # import pdb; pdb.set_trace()
        # choose the right tokenizer
        if "bert-large-cased" in path:
            tokenizer = tokenizers['bert-large-cased']
        elif "roberta-large" in path:
            tokenizer = tokenizers['roberta-large']
        elif "bert-base-cased" in path: # do this last so oracle/moe capture the correct thing first
            tokenizer = tokenizers['bert-base-cased']
        else:
            continue

        try:
            with open(f"{path}/predictions.json") as f:
                predictions = json.load(f)
        except json.decoder.JSONDecodeError:
            print(f"Error decoding json from: {path}")
            continue

        # convert objects to idxs

        # obj_ids = tokenizer.tokenize([f' {p}'] for p in predictions['x_ts'])
        #  tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x)) for x in predictions['x_ts']
        # import pdb; pdb.set_trace()
        objs_uniq = {x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' ' + x))[0] for x in set(predictions['x_ts'])}
        objs = [objs_uniq[x] for x in predictions['x_ts']]
        predictions['x_ts'] = objs

        if 'metrics' in predictions:
            predictions['RR'] = [m['RR'] for m in predictions['metrics']]
            predictions['RR_vcb'] = [m['RR_vcb'] for m in predictions['metrics']]
            del predictions['metrics']

        with open(f"{new_out_dir}/predictions.json", "w") as f:
            json.dump(predictions, f)
        shutil.copy(f"{path}/results.json", f"{new_out_dir}/results.json")
        try:
            shutil.copy(f"{path}/config.yaml", f"{new_out_dir}/config.yaml")
        except FileNotFoundError:
            pass
        # break



if __name__ == "__main__":
    main()
