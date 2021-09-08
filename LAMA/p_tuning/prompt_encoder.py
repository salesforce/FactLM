import torch
import torch.nn as nn


class PromptEncoder(torch.nn.Module):
    def __init__(self, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        template = eval(args.model.template)
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.model.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds

###### BEN'S EDITS BELOW ######

class PromptEnsembleEncoder(torch.nn.Module):
    def __init__(self, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        template = eval(args.model.template)

        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        self.ensemble_size = self.args.model.ensemble_size
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)
        # self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)

        # self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        embeddings_size = (self.ensemble_size, len(self.cloze_mask[0]), self.hidden_size)
        embeddings = torch.randn(*embeddings_size, device=self.device, requires_grad=True)
        self.register_parameter(name='embeddings', param=torch.nn.Parameter(embeddings))

        mixture_weights = 0.1 * torch.randn(self.ensemble_size, device=self.device, requires_grad=True) # initialize to be (almost) uniform and ~ magnitude 1
        self.register_parameter(name='mixture_weights', param=torch.nn.Parameter(mixture_weights))

        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.model.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)

        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))

    def forward(self):
        input_embeds = self.embeddings # (self.seq_indices)
        output_embeds = torch.matmul(torch.softmax(self.mixture_weights, dim=0), self.embeddings.reshape(self.ensemble_size, -1)).reshape(-1, self.hidden_size)
        output_embeds = self.mlp_head(self.lstm_head(output_embeds.unsqueeze(0))[0]).squeeze()
        return output_embeds



def create_prompt_encoder(args):
    if args.model.adapter.name == 'identity':
        return PromptEncoderIdentity
    elif args.model.adapter.name == 'prefix-lstm':
        return PromptEncoderPrefixLSTM
    elif args.model.adapter.name == 'rewrite-lstm':
        return PromptEncoderRewriteLSTM
    elif args.model.adapter.name == 'rewrite-ptuning':
        return PromptEncoderPtuningInputConditional
    elif args.model.adapter.name == 'rewrite-ptuning-mo':
        return PromptEncoderPtuningMaskOnlyInputConditional
    elif args.model.adapter.name == 'rewrite-ptuning-so':
        return PromptEncoderPtuningSubjectOnlyInputConditional
    elif args.model.adapter.name == 'rewrite-trans':
        return PromptEncoderRewriteTransformer

class AbstractPromptEncoder(torch.nn.Module):
    def __init__(self, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        # self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args

        self._name = None

    @property
    def name(self):
        return self._name

    def tokenize(self, query, prompt_tokens):
        raise NotImplementedError("Override AbstractPromptEncoder class method")

    def forward(self, tokens=None, embeddings=None):
        raise NotImplementedError("Override AbstractPromptEncoder class method")


class PromptEncoderIdentity(AbstractPromptEncoder):
    def __init__(self, hidden_size, tokenizer, device, args):
        super().__init__(hidden_size, tokenizer, device, args)

    def tokenize(self, query, prompt_tokens):
        return self.tokenizer(' ' + query)['input_ids']

    def forward(self, tokens=None, embeddings=None):
        return embeddings


class PromptEncoderPrefixLSTM(AbstractPromptEncoder):
    def __init__(self, hidden_size, tokenizer, device, args):
        super().__init__(hidden_size, tokenizer, device, args)
        self._name = "prefix-lstm"
        self.cloze_length = self.args.model.adapter.prefix_len

        # init model
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.model.adapter.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.cloze_length * self.hidden_size))


    def tokenize(self, query, prompt_tokens):
        token_ids = self.tokenizer(' ' + query)['input_ids']
        # if token_ids[0] == self.tokenizer.cls_token_id:
        # import pdb; pdb.set_trace()
        return prompt_tokens * self.cloze_length + token_ids

    def forward(self, tokens=None, embeddings=None):
        if self.args.model.distributed:
            self.lstm_head.flatten_parameters()
        # remove the prompt token embeddings so we can replace them
        bsz, seqlen, embed_dim = embeddings.shape
        input_embeddings = embeddings[:, self.cloze_length:, :] # bsz, seqlen, embed_dim

        # run the lstm, pool the output, and upproject it so we have the right number of embeddings
        # import pdb; pdb.set_trace()
        lstm_output = self.lstm_head(input_embeddings)[0]
        output_embeddings = self.mlp_head(torch.max(lstm_output, dim=1).values) # max pool
        output_embeddings = output_embeddings.reshape(bsz, self.cloze_length, self.hidden_size)
        # embeddings[:, :self.cloze_length] = output_embeddings
        result_embeddings = torch.cat((output_embeddings, embeddings[:, self.cloze_length:]), dim=1) #  = output_embeddings

        return result_embeddings


class PromptEncoderRewriteLSTM(AbstractPromptEncoder):
    """
    Rewrites the human-written prompt in a continuous space:
    e.g. X -> X', BERT(X')
    """
    def __init__(self, hidden_size, tokenizer, device, args):
        super().__init__(hidden_size, tokenizer, device, args)
        self._name = "rewrite-lstm"
        # self.cloze_length = self.args.model.adapter.prefix_len
        self.tokenizer = tokenizer

        # init model
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.model.adapter.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Linear(self.hidden_size, self.hidden_size)


    def tokenize(self, query, prompt_tokens):
        token_ids = self.tokenizer(' ' + query)['input_ids']
        return token_ids

    def forward(self, tokens=None, embeddings=None):
        # if self.args.model.distributed:
        #     self.lstm_head.flatten_parameters()
        # lstm_output = self.mlp_head(self.lstm_head(embeddings)[0])

        attention_mask = tokens != self.tokenizer.pad_token_id
        # remove the prompt token embeddings so we can replace them
        # bsz, seqlen, embed_dim = embeddings.shape
        # run the lstm, pool the output, and upproject it so we have the right number of embeddings
        # embeds = self.transformer(inputs_embeds=embeddings, attention_mask=attention_mask, output_hidden_states=True)[-1]
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeddings, attention_mask.sum(dim=1), batch_first=True, enforce_sorted=False)
        lstm_output = self.mlp_head(
                nn.utils.rnn.pad_packed_sequence(
                        self.lstm_head(packed_embeds)[0],
                        batch_first=True
                    )[0]
                )
        return lstm_output

        # output_embeddings = self.mlp_head(torch.max(lstm_output, dim=1).values)
        # output_embeddings = output_embeddings.reshape(bsz, self.cloze_length, self.hidden_size)
        # embeddings[:, :self.cloze_length] = output_embeddings
        # result_embeddings = torch.cat((output_embeddings, embeddings[:, self.cloze_length:]), dim=1) #  = output_embeddings
        # return result_embeddings


class PromptEncoderPtuningInputConditional(AbstractPromptEncoder):
    """
    Unlike the P-tuning prompt encoder, update the embeddings based on the natural language prompt input
    """
    def __init__(self, hidden_size, tokenizer, device, args):
        super().__init__(hidden_size, tokenizer, device, args)
        self._name = "rewrite-ptuning"
        self.device = args.device
        self.tokenizer = tokenizer
        self.template = eval(args.model.template)
        self.spell_length = sum(self.template)
        # ent embedding
        self.cloze_length = self.template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.model.adapter.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.spell_length * self.hidden_size))
        print("init prompt encoder...")

        self.mlm_embeddings = None
        self.mlm_embeddings_cache = []

    def tokenize(self, query, prompt_tokens, x_h):
        # token_ids = self.tokenizer(' ' + query, return_tensors="pt")['input_ids'].to(self.device)
        self.mlm_embeddings_cache.append(query)
        # with torch.no_grad():
        #     self.mlm_embeddings_cache.append(self.mlm_embeddings(token_ids))
        # return None
        return [[self.tokenizer.cls_token_id]  # [CLS]
                + prompt_tokens * self.template[0]
                + [self.tokenizer.mask_token_id]  # head entity
                + prompt_tokens * self.template[1]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
                + (prompt_tokens * self.template[2] if self.template[
                                                           2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                + [self.tokenizer.sep_token_id]
                ]

    def forward(self, tokens=None, embeddings=None):
        # input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        # lstm_output = self.lstm_head(input_embeddings)[0]
        # import pdb; pdb.set_trace()
        # original_token_ids = self.tokenizer([' ' + query for query in self.mlm_embeddings_cache], return_tensors="pt", padding=True)['input_ids'].to(self.device)
        query = self.mlm_embeddings_cache.pop(0)
        original_token_ids = self.tokenizer(' ' + query, return_tensors="pt")['input_ids'].to(self.device)
        bsz = len(original_token_ids)
        with torch.no_grad():
            mlm_embeddings = self.mlm_embeddings(original_token_ids)
        lstm_output = self.lstm_head(mlm_embeddings)[0]
        output_embeddings = self.mlp_head(torch.max(lstm_output, dim=1).values) # max pool
        output_embeddings = output_embeddings.reshape(bsz, self.spell_length, self.hidden_size)

        #output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeddings.squeeze()

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if "embeddings" in name:
                continue
            yield param


class PromptEncoderRewriteTransformer(AbstractPromptEncoder):
    """
    Rewrites the human-written prompt in a continuous space using the pretrained transformer:
    e.g. X -> X', BERT(X')
    """
    def __init__(self, hidden_size, tokenizer, device, args):
        super().__init__(hidden_size, tokenizer, device, args)
        self._name = "rewrite-trans"
        self.tokenizer = tokenizer

        # init model
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.model.adapter.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)

        self.mlp_head = nn.Linear(self.hidden_size, self.hidden_size)

        self.transformer = None


    def tokenize(self, query, prompt_tokens):
        token_ids = self.tokenizer(' ' + query)['input_ids']
        return token_ids

    def forward(self, tokens=None, embeddings=None):
        if self.args.model.distributed:
            self.lstm_head.flatten_parameters()

        attention_mask = tokens != self.tokenizer.pad_token_id
        # remove the prompt token embeddings so we can replace them
        # bsz, seqlen, embed_dim = embeddings.shape
        # run the lstm, pool the output, and upproject it so we have the right number of embeddings
        embeds = self.transformer(inputs_embeds=embeddings, attention_mask=attention_mask, output_hidden_states=True)[-1]
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds[-1], attention_mask.sum(dim=1), batch_first=True, enforce_sorted=False)
        lstm_output = self.mlp_head(
                nn.utils.rnn.pad_packed_sequence(
                        self.lstm_head(packed_embeds)[0],
                        batch_first=True
                    )[0]
                )
        return lstm_output

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if "transformer" in name:
                continue
            yield param


class PromptEncoderPtuningMaskOnlyInputConditional(AbstractPromptEncoder):
    """
    Unlike the P-tuning prompt encoder, update the embeddings based on the natural language prompt input
    """
    def __init__(self, hidden_size, tokenizer, device, args):
        super().__init__(hidden_size, tokenizer, device, args)
        self._name = "rewrite-ptuning-mo"
        self.device = args.device
        self.template = eval(args.model.template)
        self.spell_length = sum(self.template)
        # ent embedding
        self.cloze_length = self.template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.model.adapter.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.spell_length * self.hidden_size))
        print("init prompt encoder...")

        self.mlm_embeddings = None
        self.mlm_embeddings_cache = []

    def tokenize(self, query, prompt_tokens, x_h):
        self.mlm_embeddings_cache.append(query)
        return [[self.tokenizer.cls_token_id]  # [CLS]
                + prompt_tokens * self.template[0]
                + [self.tokenizer.mask_token_id]  # head entity
                + prompt_tokens * self.template[1]
                # + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
                + (prompt_tokens * self.template[2] if self.template[
                                                           2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                + [self.tokenizer.sep_token_id]
                ]

    def forward(self, tokens=None, embeddings=None):
        query = self.mlm_embeddings_cache.pop(0)
        original_token_ids = self.tokenizer(' ' + query, return_tensors="pt")['input_ids'].to(self.device)
        bsz = len(original_token_ids)
        with torch.no_grad():
            mlm_embeddings = self.mlm_embeddings(original_token_ids)
        lstm_output = self.lstm_head(mlm_embeddings)[0]
        output_embeddings = self.mlp_head(torch.max(lstm_output, dim=1).values) # max pool
        output_embeddings = output_embeddings.reshape(bsz, self.spell_length, self.hidden_size)

        return output_embeddings.squeeze()

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if "embeddings" in name:
                continue
            yield param


class PromptEncoderPtuningSubjectOnlyInputConditional(AbstractPromptEncoder):
    """
    Unlike the P-tuning prompt encoder, update the embeddings based on the natural language prompt input
    """
    def __init__(self, hidden_size, tokenizer, device, args):
        super().__init__(hidden_size, tokenizer, device, args)
        self._name = "rewrite-ptuning-so"
        self.tokenizer = tokenizer
        self.device = args.device
        self.template = eval(args.model.template)
        self.spell_length = sum(self.template)
        # ent embedding
        self.cloze_length = self.template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.model.adapter.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.spell_length * self.hidden_size))
        print("init prompt encoder...")

        self.mlm_embeddings = None
        self.mlm_embeddings_cache = []

    def tokenize(self, query, prompt_tokens, x_h):
        self.mlm_embeddings_cache.append(query)
        return [[self.tokenizer.cls_token_id]  # [CLS]
                + prompt_tokens * self.template[0]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
                # + [self.tokenizer.mask_token_id]  # head entity
                + prompt_tokens * self.template[1]
                # + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
                + (prompt_tokens * self.template[2] if self.template[
                                                           2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                + [self.tokenizer.sep_token_id]
                ]

    def forward(self, tokens=None, embeddings=None):
        query = self.mlm_embeddings_cache.pop(0)
        original_token_ids = self.tokenizer(' ' + query, return_tensors="pt")['input_ids'].to(self.device)
        bsz = len(original_token_ids)
        with torch.no_grad():
            mlm_embeddings = self.mlm_embeddings(original_token_ids)
        lstm_output = self.lstm_head(mlm_embeddings)[0]
        output_embeddings = self.mlp_head(torch.max(lstm_output, dim=1).values) # max pool
        output_embeddings = output_embeddings.reshape(bsz, self.spell_length, self.hidden_size)

        return output_embeddings.squeeze()

    def parameters(self, recurse=True):
        for name, param in self.named_parameters(recurse=recurse):
            if "embeddings" in name:
                continue
            yield param
