from argparse import ArgumentParser
import os
import shlex
import subprocess


TRAIN_STR_TEMPLATE = "python LAMA/cli.py --model_name {model_name} --use_original_template True --relation_id {relation_id} --vocab_strategy shared"
# TRAIN_STR_TEMPLATE = "python LAMA/cli.py --model_name {model_name} --relation_id {relation_id} --vocab_strategy shared"
ADAPTIVE_PROMPT_STR_TEMPLATE = " --use_adaptive_prompt True --prompt_encoder_name {prompt_encoder_name}"
EVAL_STR_TEMPLATE = TRAIN_STR_TEMPLATE + " --only_evaluate True"
dummy_train_str = f"echo '{TRAIN_STR_TEMPLATE}'"

LOG_DIR_TEMPLATE = "out/LAMA/prompt_model/mn_{model_name}-pe_{prompt_encoder_name}"


# seen_relations = ['P19', 'P1303', 'P264', 'P39', 'P463', 'P136', 'P31', 'P1376', 'P361', 'P37', 'P178', 'P127', 'P138', 'P20', 'P937', 'P279', 'P106', 'P47', 'P176', 'P131']
seen_relations = []

def main(args):
    # ensure log_dir exists
    log_dir = LOG_DIR_TEMPLATE.format(model_name=args.model_name, prompt_encoder_name=args.prompt_encoder_name)
    if args.notes is not None:
        log_dir += f"-notes_{args.notes}"
    os.makedirs(log_dir, exist_ok=True)

    cmd_args = {"model_name": args.model_name}
    train_str = TRAIN_STR_TEMPLATE
    eval_str = EVAL_STR_TEMPLATE
    if args.prompt_encoder_name is not None:
        train_str = train_str + ADAPTIVE_PROMPT_STR_TEMPLATE
        eval_str = eval_str + ADAPTIVE_PROMPT_STR_TEMPLATE
        cmd_args["prompt_encoder_name"] = args.prompt_encoder_name


    for relation_id in os.listdir("data/LAMA/fact-retrieval/original/"):
        if relation_id in seen_relations:
            continue

        cmd_args["relation_id"] = relation_id

        if not args.only_evaluate:
            outfilename = os.path.join(log_dir, f"{relation_id}_train.out")
            with open(outfilename, "w") as f:
                cmd = train_str.format(**cmd_args)
                print(cmd)
                subprocess.call(shlex.split(cmd), stdout=f, stderr=subprocess.STDOUT)

        if not args.only_train:
            outfilename = os.path.join(log_dir, "eval_all.out")
            with open(outfilename, "a") as f:
                cmd = eval_str.format(**cmd_args) # model_name=args.model_name, relation_id=relation_id)
                # cmd = dummy_train_str.format(**cmd_args)
                print(cmd)
                subprocess.call(shlex.split(cmd), stdout=f, stderr=subprocess.STDOUT)



if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("--only_evaluate", action='store_true')
    argparser.add_argument("--only_train", action='store_true')
    argparser.add_argument("--model_name", type=str, default="bert-base-cased")
    argparser.add_argument("--prompt_encoder_name", type=str, default=None)
    argparser.add_argument("--notes", type=str, default=None)
    args = argparser.parse_args()
    main(args)
