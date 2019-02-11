# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)

""" Pretrain transformer with Masked LM and Sentence Classification """

from random import randint, shuffle
from random import random as rand
import fire

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

import tokenization
import models
import optim
import train

from utils import set_seeds, get_device, get_random_word, truncate_tokens_pair
from torch.utils.data import Dataset, DataLoader
import csv
import itertools

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.

class SentPairDataLoader():
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, file, batch_size, tokenize, max_len, pipeline=[]):
        super().__init__()
        self.f_pos = open(file, "r", encoding='utf-8', errors='ignore')
        self.lines = csv.reader(self.f_pos, delimiter='\t', quotechar=None)
        self.tokenize = tokenize # tokenize function
        self.max_len = max_len # maximum length of tokens
        self.pipeline = pipeline
        self.batch_size = batch_size

    def __iter__(self): # iterator to load data
        batch = []
        count = 0
        for line in itertools.islice(self.lines, 1, None):
            is_next = int(line[0])
            tokens_a = self.tokenize(line[1])
            tokens_b = self.tokenize(line[2])
            truncate_tokens_pair(tokens_a, tokens_b, self.max_len)
            instance = (is_next, tokens_a, tokens_b)
            for proc in self.pipeline:
                instance = proc(instance)

            batch.append(instance)
            count+=1

            if count == self.batch_size:
                batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
                yield batch_tensors

                count = 0
                batch = []

        self.f_pos.seek(0)

class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()

    def __call__(self, instance):
        raise NotImplementedError


class Preprocess4Pretrain(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.vocab_words = vocab_words # vocabulary (sub)words
        self.indexer = indexer # function from token to token index
        self.max_len = max_len

    def __call__(self, instance):
        is_next, tokens_a, tokens_b = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)
        input_mask = [1]*len(tokens)

        # For masked Language Models
        masked_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(tokens)*self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = [i for i, token in enumerate(tokens)
                    if tokens != '[CLS]' and tokens != '[SEP]']
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if rand() < 0.8: # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5: # 10%
                tokens[pos] = get_random_word(self.vocab_words)
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer(tokens)
        masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*n_pad)
        segment_ids.extend([0]*n_pad)
        input_mask.extend([0]*n_pad)

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
            masked_weights.extend([0]*n_pad)

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)


class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.transformer = models.Transformer(cfg)

        #logits_sentence_clsf
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(cfg.dim, 2)

        #logits_paragraph_clsf
        '''
        self.fc = nn.Linear(cfg.dim, 2)
        self.activ1 = nn.Tanh()
        self.norm1 = models.LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.max_len * 2, 2)
        '''

        #logits_lm
        self.linear = nn.Linear(cfg.dim, cfg.dim)
        self.activ2 = models.gelu
        self.norm2 = models.LayerNorm(cfg)
        # decoder is shared with embedding layer
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

        #logits_same
        self.linear2 = nn.Linear(cfg.dim, cfg.vocab_size)
        #self.activ3 = models.gelu
        #self.norm3 = models.LayerNorm(cfg)
        # decoder is shared with embedding layer
        #embed_weight2 = self.transformer.embed.tok_embed.weight
        #n_vocab, n_dim = embed_weight2.size()
        #self.decoder2 = nn.Conv2d(n_dim, n_vocab, kernel=3, stride=1, padding=1)
        #self.decoder2 = nn.Linear(n_dim, n_vocab, bias=False)
        #self.decoder2.weight = embed_weight2
        #self.decoder_bias2 = nn.Parameter(torch.zeros(n_vocab))


    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)

        #logits_clsf
        pooled_h = self.activ1(self.fc(h[:, 0]))
        logits_clsf = self.classifier(pooled_h)

        #logits_paragraph_clsf
        '''
        batch_size, seq_length, hidden_size = h.size()
        reshape_h = h.view(batch_size*seq_length, hidden_size)
        pooled_h = self.activ1(self.fc(self.norm1(reshape_h)))
        pooled_h = pooled_h.view(batch_size, seq_length*2)
        logits_clsf = self.classifier(self.drop(pooled_h))
        '''

        #logits_lm
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm2(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias

        #logits_same
        logits_same = self.linear2(h)
        #h_all = self.norm3(self.activ3(self.linear2(h)))
        #logits_same = self.decoder2(h_all)

        return logits_lm, logits_clsf, logits_same


def main(train_cfg='config/pretrain.json',
         model_cfg='config/bert_base.json',
         data_file='../tbc/books_large_all.txt',
         model_file=None,
         data_parallel=True,
         vocab='../uncased_L-12_H-768_A-12/vocab.txt',
         save_dir='../exp/bert/pretrain',
         log_dir='../exp/bert/pretrain/runs',
         max_len=512,
         max_pred=20,
         mask_prob=0.15):

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
    tokenize = lambda x: tokenizer.tokenize(tokenizer.convert_to_unicode(x))

    pipeline = [Preprocess4Pretrain(max_pred,
                                    mask_prob,
                                    list(tokenizer.vocab.keys()),
                                    tokenizer.convert_tokens_to_ids,
                                    max_len)]
    data_iter = SentPairDataLoader(data_file,
                                   cfg.batch_size,
                                   tokenize,
                                   max_len,
                                   pipeline=pipeline)

    model = BertModel4Pretrain(model_cfg)
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss()
    criterion3 = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.optim4GPU(cfg, model)
    trainer = train.Trainer(cfg, model, data_iter, optimizer, save_dir, get_device())

    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    def get_loss(model, batch, global_step): # make sure loss is tensor
        input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next = batch

        #logits_lm, logits_clsf = model(input_ids, segment_ids, input_mask, masked_pos)
        logits_lm, logits_clsf, logits_same = model(input_ids, segment_ids, input_mask, masked_pos)
        loss_lm = criterion1(logits_lm.transpose(1, 2), masked_ids) # for masked LM
        loss_lm = (loss_lm*masked_weights.float()).mean()
        loss_clsf = criterion2(logits_clsf, is_next) # for sentence classification
        loss_same = criterion3(logits_same.transpose(1, 2), input_ids)
        loss_same = (loss_same*input_mask.float()).mean()
        #loss_same = loss_same.mean()
        #loss_clsf *= 1000
        print(loss_lm.item(), loss_clsf.item(), loss_same.item())
        writer.add_scalars('data/scalar_group',
                           {'loss_lm': loss_lm.item(),
                            'loss_clsf': loss_clsf.item(),
                            'loss_same': loss_same.item(),
                            'loss_total': (loss_clsf + loss_lm + loss_same).item(),
                            'lr': optimizer.get_lr()[0],
                           },
                           global_step)

        return loss_lm + loss_clsf + logits_same

    trainer.train(get_loss, model_file, None, data_parallel)


if __name__ == '__main__':
    fire.Fire(main)
