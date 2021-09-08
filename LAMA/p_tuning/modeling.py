import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
import os

import re

from transformers import AutoTokenizer

from LAMA.p_tuning.models import get_embedding_layer, create_model
from LAMA.data_utils.vocab import get_vocab_by_strategy, token_wrapper
from LAMA.data_utils.dataset import load_relations_templates, load_relations
from LAMA.p_tuning.prompt_encoder import PromptEncoder, create_prompt_encoder, PromptEnsembleEncoder


class PTuneForLAMA(torch.nn.Module):

    def __init__(self, args, device, relclf_model=None):
        super().__init__()
        self.args = args
        self.device = device
        # self.using_relclf = relclf_model is not None
        # self.using_adaptive_prompt = self.args.model.get("adapter") is not None
        self.using_template = self.args.model.get("template") is not None
        # assert not self.using_relclf or not self.args.use_adaptive_prompt, "Currently cannot use adaptive prompt with the relation classifier for prompt selection"

        # load relation templates
        # self.relation_templates = load_relations_templates(join(self.args.data_dir, self.args.relation_templates))
        self.relation_templates = load_relations_templates(self.args.data.train.template_path)

        # load tokenizer
        tokenizer_src = 'roberta-large' if 'megatron' in self.args.model.name else self.args.model.name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=False)

        # load pre-trained model
        # if 'megatron' in self.args.model.name and self.args.use_lm_finetune:
        #     raise RuntimeError("Can not apply args.use_lm_finetune=True on MegatronLM 11B.")
        self.model = create_model(self.args)

        print(self.device)
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = self.args.model.finetune
        self.embeddings = get_embedding_layer(self.args, self.model)

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))
        self.allowed_vocab_ids_tensor = torch.tensor(list(self.allowed_vocab_ids), device=self.device)

        if self.using_template: #  or self.using_relclf:
            template = eval(self.args.model.template)
            if 'gpt' in self.args.model.name or 'megatron' in self.args.model.name:
                template = (template[0], template[1], 0)
            self.template = template
            self.spell_length = sum(self.template)

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': [self.args.model.pseudo_token]})
        self.pseudo_token_id = self.tokenizer.get_vocab()[self.args.model.pseudo_token]
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        if self.args.model.type == "adapter" :# self.using_adaptive_prompt:
            self.prompt_encoder = create_prompt_encoder(self.args)(self.hidden_size, self.tokenizer, self.device, args)
            if self.args.model.adapter.name in ("rewrite-ptuning", "rewrite-ptuning-mo"):
                self.prompt_encoder.mlm_embeddings = self.embeddings
            elif self.args.model.adapter.name == "rewrite-trans":
                self.prompt_encoder.transformer = self.model
            self.prompt_encoder = self.prompt_encoder.to(self.device)
        elif self.args.model.type in ("moe", "oracle"): # , "configs/model/oracle-ensemble:
            print("LOADING PROMPT ENCODERS")
            self.prompt_encoder = self.load_all_prompt_encoders()
            self.relclf = relclf_model
        elif self.args.model.type == "p-tuning": # P-tuning
            self.prompt_encoder = PromptEncoder(self.hidden_size, self.tokenizer, self.device, args)
            self.prompt_encoder = self.prompt_encoder.to(self.device)
        elif self.args.model.type == "ensemble":
            self.prompt_encoder = PromptEnsembleEncoder(self.hidden_size, self.tokenizer, self.device, args)
            self.prompt_encoder = self.prompt_encoder.to(self.device)
        else: # baseline: don't do anything
            pass


    def load_all_prompt_encoders(self):
        prompt_dict = {}
        train_rel_id = self.args.data.train.relations_id
        relation_ids = load_relations(self.args.data.train.relations_path)
        for relation_id in relation_ids:


        # path = join(self.args.relclf_prompt_path, "{}")
        # for relation in self.relation_templates:
            # path = os.path.join()
            self.args.data.train.relations_id = relation_id
            ptuning_id = f"{self.args.model.ptuning_id}-{self.args.data.train.id}"
            path = join(self.args.model.ptuning_out_path_prefix, ptuning_id, "checkpoints")
            # print(path)
            # import pdb; pdb.set_trace()
            try:
                ckpt = self.load_newest_prompt(path)
                if ckpt is None:
                    raise FileNotFoundError("No checkpoints found")
            except FileNotFoundError:
                print(f"Unable to find prompt for relation {relation_id}")
                continue
            if self.args.model.ptuning_type == "p-tuning":
                prompt_dict[relation_id] = PromptEncoder(self.hidden_size, self.tokenizer, self.device, self.args)
            elif self.args.model.ptuning_type == "ensemble":
                prompt_dict[relation_id] = PromptEnsembleEncoder(self.hidden_size, self.tokenizer, self.device, self.args)
            prompt_dict[relation_id].load_state_dict(
                ckpt['embedding']
            )
            # print(f"Loading {relation_id}")
            prompt_dict[relation_id] = prompt_dict[relation_id].to(self.device)

        # restore train relations path
        # self.args.data.train.relations_path = train_rel_path
        self.args.data.train.relations_id = train_rel_id
        assert prompt_dict
        return prompt_dict

    def load_newest_prompt(self, path):
        most_recent_ckpt = None
        for ckpt_name in os.listdir(path):
            # choose the most recent one I guess
            ckpt = torch.load(join(path, ckpt_name), map_location=self.device)
            most_recent_ckpt = ckpt if (
                    most_recent_ckpt is None or ckpt['time'] > most_recent_ckpt['time']
            ) else most_recent_ckpt
        return most_recent_ckpt

    def embed_input(self, queries, predicted_relations=None):
        """Only use predicted_relations when using the relation classifier"""
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        # import pdb; pdb.set_trace()
        # For using handcraft prompts
        # if self.args.use_original_template and not self.args.use_adaptive_prompt and not self.using_relclf:
        if self.args.model.type in ("baseline",): # self.args.use_original_template and not self.args.use_adaptive_prompt and not self.using_relclf:
            return raw_embeds

        if self.args.model.type in ("adapter",) and self.args.model.adapter.name not in ("rewrite-ptuning", "rewrite-ptuning-mo"): # use_adaptive_prompt:
            return self.prompt_encoder(tokens=queries_for_embedding, embeddings=raw_embeds)

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        if self.args.model.type in ("moe", "oracle") or (self.args.model.type == "adapter" and self.args.model.adapter.name in ("rewrite-ptuning", "rewrite-ptuning-mo")): # self.using_relclf:
            # get the predictions
            # predicted_relations = self.relclf.predict(self.tokenizer.decode())
            # import pdb; pdb.set_trace()
            for bidx in range(bz):
                if self.args.model.type == "adapter":
                    replace_embeds = self.prompt_encoder(tokens=queries_for_embedding[bidx], embeddings=raw_embeds[bidx])
                else:
                    replace_embeds = self.prompt_encoder[predicted_relations[bidx]]()
                for i in range(self.spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        else:
            # When we aren't using relclf, we only have a single prompt encoder, so we only have to do the forward pass
            # a single time
            replace_embeds = self.prompt_encoder()
            for bidx in range(bz):
                for i in range(self.prompt_encoder.spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds

    def get_query(self, x_h, prompt_tokens, template, x_t=None):
        # For using hand-written prompts
        # if not self.args.model.use_original_template:
        if not self.using_template:
            # default
            if 'gpt' in self.args.model.name or 'megatron' in self.args.model.name:
                # query = re.sub(r'\[Y\].*', '', self.relation_templates[self.args.relation_id].replace('[X]', x_h))
                # query = re.sub(r'\[Y\].*', '', self.relation_templates[relation].replace('[X]', x_h))
                query = re.sub(r'\[Y\].*', '', template.replace('[X]', x_h))
                if self.args.model.type == "adapter": # self.using_adaptive_prompt:
                    return self.prompt_encoder.tokenize(query, prompt_tokens)
                else:
                    return self.tokenizer(' ' + query)['input_ids']
            else:
                # query = self.relation_templates[self.args.relation_id].replace('[X]', x_h).replace('[Y]', self.tokenizer.mask_token)
                # query = self.relation_templates[relation].replace('[X]', x_h).replace('[Y]', self.tokenizer.mask_token)
                query = template.replace('[X]', x_h).replace('[Y]', self.tokenizer.mask_token)
                # if self.using_adaptive_prompt:
                if self.args.model.type == "adapter":  # self.using_adaptive_prompt:
                    return self.prompt_encoder.tokenize(query, prompt_tokens)
                else:
                    return self.tokenizer(' ' + query)['input_ids']

        # For P-tuning
        if 'gpt' not in self.args.model.name and 'megatron' not in self.args.model.name:
            # BERT-style model
            if self.args.model.type == "adapter" and self.args.model.adapter.name in ("rewrite-ptuning", "rewrite-ptuning-mo"):
                query = template.replace('[X]', x_h).replace('[Y]', self.tokenizer.mask_token)
                return self.prompt_encoder.tokenize(query, prompt_tokens, x_h) # store the embeddings
            return [[self.tokenizer.cls_token_id]  # [CLS]
                    + prompt_tokens * self.template[0]
                    + [self.tokenizer.mask_token_id]  # head entity
                    + prompt_tokens * self.template[1]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
                    + (prompt_tokens * self.template[2] if self.template[
                                                               2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
                    + [self.tokenizer.sep_token_id]
                    ]
        elif 'gpt' in self.args.model.name or 'megatron' in self.args.model.name:
            # GPT-style models
            return [prompt_tokens * self.template[0]
                    + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # head entity
                    + prompt_tokens * self.template[1]
                    + (self.tokenizer.convert_tokens_to_ids(
                self.tokenizer.tokenize(' ' + x_t)) if x_t is not None else [])
                    ]
        else:
            raise NotImplementedError("The query template for {} has not been defined.".format(self.args.model_name))

    def get_predicted_relations(self, x_hs, relations, templates):
        """Currently, we only support bidirectional models (bert/roberta)
        Args:
            x_hs: <pass>
            relations: str: These are the gold relations, which are used for getting the prompts
             (which are then used to predict the relation)
        """
        # return ["P1001" for _ in range(len(relations))]
        if self.args.model.type == "oracle":# relclf_use_gold_relations:
            return relations
        # queries = [self.relation_templates[relation].replace('[X]', x_h).replace('[Y]', self.tokenizer.mask_token)
        #             for x_h, relation in zip(x_hs, relations)]
        # import pdb; pdb.set_trace()
        queries = [template.replace('[X]', x_h).replace('[Y]', self.tokenizer.mask_token)
                   for x_h, template in zip(x_hs, templates)]
        predicted_relations = self.relclf.predict(queries)
        return predicted_relations


    def calculate_metrics(self, pred_ids, label_id):
        metrics = {} # defaultdict(list)
        metrics["P@1"] = (pred_ids[0] == label_id).float().item()
        metrics["P@5"] = (pred_ids[:5] == label_id).sum().item()
        metrics["P@10"] = (pred_ids[:10] == label_id).sum().item()
        metrics["P@100"] = (pred_ids[:100] == label_id).sum().item()
        metrics["RR"] = 1 / (1 + torch.where(pred_ids == label_id)[0].item())
        metrics["top10"] = pred_ids[:10].tolist()
        return metrics

    def forward(self, x_hs, x_ts, relations, templates, return_candidates=False):
        bz = len(x_hs)

        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
        x_ts = [token_wrapper(self.args, x_t) for x_t in x_ts]
        queries = [torch.LongTensor(self.get_query(x_hs[i], prompt_tokens, templates[i])).squeeze(0) for i in range(bz)]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
        predicted_relations = self.get_predicted_relations(x_hs, relations, templates) if self.args.model.type in ("oracle", "moe") else None
        # construct label ids
        label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape(
            (bz, -1)).to(self.device)
        attention_mask = queries != self.pad_token_id

        # get embedded input
        inputs_embeds = self.embed_input(queries, predicted_relations=predicted_relations)

        def bert_out():
            label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(
                1).to(self.device)  # bz * 1
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            labels = labels.scatter_(1, label_mask, label_ids)
            output = self.model(inputs_embeds=inputs_embeds.to(self.device),
                                attention_mask=attention_mask.to(self.device).bool(),
                                labels=labels.to(self.device))
            # loss, logits = output.loss, output.logits
            loss, logits = output

            # pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            batch_metrics = []
            top10 = []

            for i in range(bz):
                logit = logits[i][label_mask[i, 0]]
                sorted_idxs = torch.argsort(logit, descending=True)
                metrics = self.calculate_metrics(sorted_idxs, label_ids[i]) # pred_ids = torch.argsort(logit)

                # vcb means we only consider the vocab items that are in self.allowed_vocab_ids
                # logit_vcb = torch.gather(logit, 0, self.allowed_vocab_ids_tensor)
                # (torch.where(sorted_logit == self.allowed_vocab_ids_tensor.unsqueeze(1), sorted_logit, torch.tensor(-1)) + 1).sum(dim=0) - 1
                # sorted_logit_vcb = sorted_idxs_vcb[sorted_logit_vcb >= 0]

                sorted_idxs_vcb = self.allowed_vocab_ids_tensor[torch.argsort(logit[self.allowed_vocab_ids_tensor], descending=True)]
                # label_vcb = torch.where(self.allowed_vocab_ids_tensor[sorted_idxs_vcb] == label_ids[i])[0]
                # https://discuss.pytorch.org/t/any-way-of-filtering-given-rows-from-a-tensor-a/83828/2
                # min_dist = torch.min(torch.cdist(sorted_idxs.float(), self.allowed_vocab_ids_tensor.float())).values
                # sorted_idxs_vcb = sorted_idxs[min_dist].int()
                # sorted_idxs_vcb = torch.tensor([idx for idx in sorted_idxs if idx.item() in self.allowed_vocab_ids], device=self.device)
                # assert label_ids[i] in sorted_idxs_vcb
                # assert label_vcb in sorted_idxs_vcb
                metrics_vcb = self.calculate_metrics(sorted_idxs_vcb, label_ids[i])
                # metrics_vcb = self.calculate_metrics(sorted_idxs_vcb, label_vcb)
                metrics_vcb = {f"{metric_name}_vcb": value for metric_name, value in metrics_vcb.items()}
                batch_metrics.append(dict(**metrics, **metrics_vcb))
                hit1 += metrics_vcb["P@1_vcb"]
                top10.append(metrics_vcb.pop("top10_vcb"))


            # for i in range(bz):
            #     pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
            #     for pred in pred_seq:
            #         if pred in self.allowed_vocab_ids:
            #             break
            #     if pred == label_ids[i, 0]:
            #         hit1 += 1
            #     top10.append(pred)

            if return_candidates:
                return loss, hit1, batch_metrics, top10
            if self.args.model.distributed:
                return torch.tensor((loss, hit1)).to(self.device)
            return loss, hit1, batch_metrics

        def gpt_out():
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1).to(self.device)
            labels = labels.scatter_(1, label_mask, label_ids)

            output = self.model(inputs_embeds=inputs_embeds.to(self.device).half(),
                                attention_mask=attention_mask.to(self.device).half(),
                                labels=labels.to(self.device))
            loss, logits = output.loss, output.logits

            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            for i in range(bz):
                top10.append([])
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        top10[-1].append(pred)
                        if len(top10[-1]) >= 10:
                            break
                pred = top10[-1][0]
                if pred == label_ids[i, 0]:
                    hit1 += 1

            if return_candidates:
                return loss, hit1, top10
            return loss, hit1

        def megatron_out():
            labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
            label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1).to(self.device)
            labels = labels.scatter_(1, label_mask, label_ids)
            if not self.args.use_lm_finetune:
                _attention_mask = attention_mask.float().half()
                _input_embeds = inputs_embeds.float().half()
            else:
                _attention_mask = attention_mask.float()
                _input_embeds = inputs_embeds.float()
            output = self.model.decoder.predict(prev_output_tokens=queries,
                                                inputs_embeds=_input_embeds.to(self.device),
                                                attention_mask=_attention_mask.to(self.device).bool(),
                                                labels=labels.to(self.device))
            logits, loss = output

            pred_ids = torch.argsort(logits, dim=2, descending=True)
            hit1 = 0
            top10 = []
            for i in range(bz):
                top10.append([])
                pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
                for pred in pred_seq:
                    if pred in self.allowed_vocab_ids:
                        top10[-1].append(pred)
                        if len(top10[-1]) >= 10:
                            break
                pred = top10[-1][0]
                if pred == label_ids[i, 0]:
                    hit1 += 1
            if return_candidates:
                return loss, hit1, top10
            return loss, hit1

        if 'bert' in self.args.model.name:
            return bert_out()
        elif 'gpt' in self.args.model.name:
            return gpt_out()
        elif 'megatron' in self.args.model.name:
            return megatron_out()
        else:
            raise NotImplementedError()
