from collections import defaultdict
from copy import copy
import json
import random

import nlpaug.augmenter.char as nac

random.seed(444)

def main():
    aug = nac.KeyboardAug(stopwords=["[X]", "[Y]"], include_upper_case=False, tokenizer=lambda x: x.split(), reverse_tokenizer=lambda x: " ".join(x))
    with open("data/templates/relations_id.json") as f:
        templates = json.load(f)

    corrupted_templates = defaultdict(list)
    for relation in templates:
        for template in templates[relation]:
            augmented_template = copy(template)
            augmented_template["template"] = aug.augment(template["template"])
            # assert "[X]" in 
            corrupted_templates[relation].append(augmented_template)
            


    with open('data/templates/relations_ood_keyboard.json', 'w') as f:
        json.dump(corrupted_templates, f)

        

if __name__ == "__main__":
    main()

