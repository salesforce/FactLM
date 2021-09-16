"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from transformers import AutoTokenizer, AutoModelForMaskedLM


def create_model(args):
    MODEL_CLASS, _ = get_model_and_tokenizer_class(args)
    model = MODEL_CLASS.from_pretrained(args.model.name)
    return model


def get_model_and_tokenizer_class(args):
    if 'bert' in args.model.name:
        return AutoModelForMaskedLM, AutoTokenizer
    else:
        raise NotImplementedError("This model type ``{}'' is not implemented.".format(args.model_name))


def get_embedding_layer(args, model):
    if 'roberta' in args.model.name:
        embeddings = model.roberta.get_input_embeddings()
    elif 'bert' in args.model.name:
        embeddings = model.bert.get_input_embeddings()
    else:
        raise NotImplementedError()
    return embeddings
