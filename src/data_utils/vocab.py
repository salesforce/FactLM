"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json
from os.path import join


def init_vocab(args):
    global lama_vocab
    lama_vocab = json.load(open(join(args.data.vocab_path)))


def token_wrapper(args, token):
    if 'roberta' in args.model.name or 'gpt' in args.model.name or 'megatron' in args.model.name:
        return 'Ġ' + token
    else:
        return token


def get_vocab(model_name, strategy):
    if 'roberta' in model_name:
        return lama_vocab['roberta-large']
    else:
        assert model_name in lama_vocab
        return lama_vocab[model_name]


def get_vocab_by_strategy(args, tokenizer):
    if args.data.vocab == 'original':
        return tokenizer.get_vocab()
    else:
        return get_vocab(args.model.name, args.data.vocab)
