from collections import defaultdict
import json
import os
import random

random.seed(415)

BASE_DIR = "/Users/benjamin.newman/Documents/syntax-effect-proj"

RELATION_FILE = os.path.join(BASE_DIR, "pararel_repo/data/trex/data/relations.jsonl")
DATA_DIRS = [
    ("pararel_repo/data/pattern_data/graphs_json", "jsonl"),
    ("pararel_repo/data/pattern_data/graphs_lexical_substitution_felicity", "jsonl"),
    ("LPAQA/prompt/manual_paraphrase", "txt"),
    ("LPAQA/prompt/mine", "jsonl"),
    ("LPAQA/prompt/paraphrase", "jsonl"),
]

def load_jsonl_file(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_txt_file(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(line.strip())
    return data

def write_jsonl_file(data, out_f):
    with open(out_f, 'w') as f:
        for obj in data:
            json.dump(obj, f)
            f.write('\n')

def write_json_file(data, out_f):
    with open(out_f, 'w') as f:
        json.dump(data, f)

def main():
    relations = load_jsonl_file(RELATION_FILE)
    template_dict = defaultdict(list)
    prev_seen_templates = set()
    for relation in relations:
        relation = relation['relation']
        for data_dir, ext in DATA_DIRS:
            template_path = os.path.join(BASE_DIR, data_dir, f"{relation}.{ext}")
            load_file_fn = load_jsonl_file if ext == "jsonl" else load_txt_file
            try:
                templates = load_file_fn(template_path)
            except FileNotFoundError:
                print(f"Unable to find relation {relation} in the directory {data_dir}")
                continue

            for template in templates:
                if ext == "txt":
                    template = {"template": template}
                template["source"] = data_dir
                template["relation"] = relation
                if "template" not in template:
                    template["template"] = template["pattern"]

                # No duplicate templates
                if template["template"] in prev_seen_templates:
                    continue
                else:
                    prev_seen_templates.add(template["template"])

                # specific filters
                if "felicity" in data_dir and not template["felicity"]:
                    continue

                template_dict[relation].append(template)
        # assert template_dict[relation]

    write_json_file(template_dict, "data/templates/relations.json")

    # and let's also create two splits
    template_dict_split_1 = defaultdict(list)
    template_dict_split_2 = defaultdict(list)
    for relation, templates in template_dict.items():
        # shuffle the templates
        shuffled_templates = random.sample(templates, len(templates))
        template_dict_split_1[relation] = shuffled_templates[len(shuffled_templates) // 4:]
        template_dict_split_2[relation] = shuffled_templates[:len(shuffled_templates) // 4]

    write_json_file(template_dict_split_1, "data/templates/relations_id.json")
    write_json_file(template_dict_split_2, "data/templates/relations_ood_human_prompts.json")




if __name__ == "__main__":
    main()
