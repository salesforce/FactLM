from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM


def create_model(args):
    if '11b' in args.model.name:
        from ..megatron_11b.megatron_wrapper import load_megatron_lm
        print("Warning: loading MegatronLM (11B) in fp16 requires about 28G GPU memory, and may need 3-5 minutes to load.")
        return load_megatron_lm(args)
    MODEL_CLASS, _ = get_model_and_tokenizer_class(args)
    model = MODEL_CLASS.from_pretrained(args.model.name)
#    if not args.use_lm_finetune:
#        if 'megatron' in args.model.name:
#            raise NotImplementedError("MegatronLM 11B is not for fine-tuning.")
#         if not args.no_cuda:
#             model = model.half()
    return model


def get_model_and_tokenizer_class(args):
    if 'gpt' in args.model.name:
        return GPT2LMHeadModel, AutoTokenizer
    elif 'bert' in args.model.name:
        return AutoModelForMaskedLM, AutoTokenizer
    elif 'megatron' in args.model.name:
        return None, AutoTokenizer
    else:
        raise NotImplementedError("This model type ``{}'' is not implemented.".format(args.model_name))


def get_embedding_layer(args, model):
    if 'roberta' in args.model.name:
        embeddings = model.roberta.get_input_embeddings()
    elif 'bert' in args.model.name:
        embeddings = model.bert.get_input_embeddings()
    elif 'gpt' in args.model.name:
        embeddings = model.base_model.get_input_embeddings()
    elif 'megatron' in args.model.name:
        embeddings = model.decoder.embed_tokens
    else:
        raise NotImplementedError()
    return embeddings
