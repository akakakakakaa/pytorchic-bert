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

from utils import set_seeds, get_device, get_random_word, truncate_tokens_pair, truncate_tokens
from torch.utils.data import Dataset, DataLoader
import csv
import itertools
import pickle

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.

class CustomVocabTokenizer():
    def __init__(self, word_vocab_file, pos_vocab_file, dep_vocab_file, pos_dep_word_vocab_file):
        self.word_tokenizer = tokenization.FullTokenizer(vocab_file=word_vocab_file, do_lower_case=True,
                                                    other_tokens=["[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        self.word_tokenize = lambda x: self.word_tokenizer.tokenize(self.word_tokenizer.convert_to_unicode(x))
        self.pos_tokenizer = tokenization.FullTokenizer(vocab_file=pos_vocab_file, do_lower_case=True,
                                                   other_tokens=["[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        self.pos_tokenize = lambda x: self.pos_tokenizer.tokenize(self.pos_tokenizer.convert_to_unicode(x))
        self.dep_tokenizer = tokenization.FullTokenizer(vocab_file=dep_vocab_file, do_lower_case=True,
                                                   other_tokens=["[UNK]", "[CLS]", "[SEP]", "[MASK]"])
        self.dep_tokenize = lambda x: self.dep_tokenizer.tokenize(self.dep_tokenizer.convert_to_unicode(x))

        with open(pos_dep_word_vocab_file, 'rb') as f:
            pos_dep_word_2d_arr = pickle.load(f)

        new_pos_dep_word_2d_arr = {}
        for pos, dep_word in pos_dep_word_2d_arr.items():
            new_pos = self.pos_tokenizer.convert_tokens_to_ids([self.convert_to_unicode(pos)])[0]
            new_pos_dep_word_2d_arr[new_pos] = {}
            for dep, words in dep_word.items():
                new_dep = self.dep_tokenizer.convert_tokens_to_ids([self.convert_to_unicode(dep)])[0]
                new_pos_dep_word_2d_arr[new_pos][new_dep] = self.word_tokenizer.convert_tokens_to_ids(self.convert_to_unicode(" ".join(words)).split())


    def tokenize(self, words, pos, dep):
        return self.word_tokenize(words), self.pos_tokenize(pos), self.dep_tokenize(dep)

    def convert_tokens_to_ids(self, words, pos, dep):
        return self.word_tokenizer.convert_tokens_to_ids(words),\
               self.pos_tokenizer.convert_tokens_to_ids(pos),\
               self.dep_tokenizer.convert_tokens_to_ids(dep)

    def convert_to_unicode(self, x):
        return self.word_tokenizer.convert_to_unicode(x)

    def get_word_vocab_size(self):
        return len(self.word_tokenizer.vocab)

    def get_pos_vocab_size(self):
        return len(self.pos_tokenizer.vocab)

    def get_dep_vocab_size(self):
        return len(self.dep_tokenizer.vocab)


class TifuDataLoader():
    """ Load sentence pair (sequential or random order) from corpus """
    def __init__(self, file, batch_size, tokenize, max_len, pipeline=[]):
        super().__init__()
        self.f_pos = open(file, "r", encoding='utf-8', errors='ignore')
        self.lines = csv.reader(self.f_pos, delimiter='\t', quotechar=None)
        self.tokenize = tokenize
        self.max_len = max_len # maximum length of tokens
        self.pipeline = pipeline
        self.batch_size = batch_size

    def __iter__(self): # iterator to load data
        batch = []
        count = 0
        for line in itertools.islice(self.lines, 1, None):
            input_tokens, input_pos, input_dep = self.tokenize(line[0], line[1], line[2])
            target_tokens, target_pos, target_dep = self.tokenize(line[3], line[4], line[5])

            instance = (input_tokens, input_pos, input_dep, target_tokens, target_pos, target_dep)
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
    def __init__(self, max_pred, mask_prob, vocab_words, vocab_pos, vocab_dep, indexer, max_len=384):
        super().__init__()
        self.max_len = max_len
        self.max_pred = max_pred # max tokens of prediction
        self.mask_prob = mask_prob # masking probability
        self.vocab_words = vocab_words # vocabulary (sub)words
        self.vocab_pos = vocab_pos
        self.vocab_dep = vocab_dep
        self.indexer = indexer # function from token to token index
        self.max_len = max_len

    def __call__(self, instance):
        input_tokens, input_pos, input_dep, target_tokens, target_pos, target_dep = instance

        # -3  for special tokens [CLS], [SEP], [SEP]
        truncate_tokens_pair(input_tokens, target_tokens, self.max_len - 3)
        truncate_tokens_pair(input_pos, target_pos, self.max_len - 3)
        truncate_tokens_pair(input_dep, target_dep, self.max_len - 3)
        target_tokens = truncate_tokens(target_tokens, self.max_len)
        target_pos = truncate_tokens(target_pos, self.max_len)
        target_dep = truncate_tokens(target_dep, self.max_len)

        # Add Special Tokens
        word_tokens = ['[CLS]'] + input_tokens + ['[SEP]'] + target_tokens + ['[SEP]']
        pos_tokens = ['[CLS]'] + input_pos + ['[SEP]'] + target_pos + ['[SEP]']
        dep_tokens = ['[CLS]'] + input_dep + ['[SEP]'] + target_dep + ['[SEP]']
        input_segment_ids = [0]*(len(input_tokens)+2) + [1]*(len(target_tokens)+1)
        input_mask = [1]*len(word_tokens)

        target_mask = [1]*len(target_tokens)

        # For masked Language Models
        masked_word_tokens, masked_pos = [], []
        # the number of prediction is sometimes less than max_pred when sequence is short
        n_pred = min(self.max_pred, max(1, int(round(len(word_tokens)*self.mask_prob))))
        # candidate positions of masked tokens
        #
        #cand_pos = [i for i, token in enumerate(word_tokens)
        #            if word_tokens != '[CLS]' and word_tokens != '[SEP]']
        #Detect SEP for summary
        cand_pos = [i for i, token in enumerate(word_tokens)
                    if word_tokens != '[CLS]']
        shuffle(cand_pos)
        for pos in cand_pos[:n_pred]:
            masked_word_tokens.append(word_tokens[pos])
            masked_pos.append(pos)
            if rand() < 0.8: # 80%
                word_tokens[pos] = '[MASK]'
        masked_weights = [1]*len(masked_word_tokens)

        #replace right as mask for summary
        #if rand() < 0.1:
        #    word_tokens = word_tokens[:len(input_tokens)+2] + ['[MASK]']*len(self.max_len - len(input_tokens)+2)
        #    pos_tokens = pos_tokens[:len(input_pos)+2] + ['[MASK]']*len(self.max_len - len(input_pos)+2)
        #    dep_tokens = dep_tokens[:len(input_dep)+2] + ['[MASK]']*len(self.max_len - len(input_dep)+2)

        input_word_ids, input_pos_ids, input_dep_ids = self.indexer(word_tokens, pos_tokens, dep_tokens)
        masked_word_ids, _, _ = self.indexer(masked_word_tokens, [], [])
        target_word_ids, target_pos_ids, target_dep_ids = self.indexer(target_tokens, target_pos, target_dep)

        # Zero Padding
        input_n_pad = self.max_len - len(input_word_ids)
        input_word_ids.extend([0]*input_n_pad)
        input_pos_ids.extend([0]*input_n_pad)
        input_dep_ids.extend([0]*input_n_pad)
        input_segment_ids.extend([0]*input_n_pad)
        input_mask.extend([0]*input_n_pad)

        target_n_pad = self.max_len - len(target_word_ids)
        target_word_ids.extend([0]*target_n_pad)
        target_pos_ids.extend([0]*target_n_pad)
        target_dep_ids.extend([0]*target_n_pad)
        target_mask.extend([0]*target_n_pad)

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_word_ids.extend([0]*n_pad)
            masked_pos.extend([0]*n_pad)
            masked_weights.extend([0]*n_pad)

        return (input_word_ids,
                input_pos_ids,
                input_dep_ids,
                input_segment_ids,
                input_mask,
                masked_word_ids,
                masked_pos,
                masked_weights,
                target_word_ids,
                target_pos_ids,
                target_dep_ids,
                target_mask)


class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg, word_vocab_size, pos_vocab_size, dep_vocab_size):
        super().__init__()
        self.transformer = models.Transformer(cfg)

        #logits_pos
        self.fc1 = nn.Linear(cfg.dim, cfg.dim)
        self.activ1 = models.gelu
        self.norm1 = models.LayerNorm(cfg)
        self.decoder1 = nn.Linear(cfg.dim, pos_vocab_size)

        #logits_vocab
        self.fc2 = nn.Linear(cfg.dim, cfg.dim)
        self.activ2 = models.gelu
        self.norm2 = models.LayerNorm(cfg)
        self.decoder2 = nn.Linear(cfg.dim, dep_vocab_size)

        #logits_word_vocab_size
        self.fc3 = nn.Linear(cfg.dim, cfg.dim)
        self.activ3 = models.gelu
        self.norm3 = models.LayerNorm(cfg)
        embed_weight = self.transformer.embed.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder3 = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder3.weight = embed_weight
        self.decoder3_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self,
                input_word_ids,
                input_segment_ids,
                masked_pos,
                input_mask,
                target_word_ids,
                target_mask):
        h = self.transformer(input_word_ids, input_segment_ids, input_mask)

        input_mask = input_mask[:, :, None].expand(-1, -1, h.size(-1))
        h_input_masked = torch.gather(h, 1, input_mask)
        h_masked_pos = self.norm1(self.activ1(self.fc1(h_input_masked)))
        logits_pos = self.decoder1(h_masked_pos)

        h_masked_dep = self.norm2(self.activ2(self.fc2(h_input_masked)))
        logits_dep = self.decoder2(h_masked_dep)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked_word = self.norm3(self.activ3(self.fc3(h_masked)))
        logits_word = self.decoder3(h_masked_word) + self.decoder3_bias

        return logits_pos, logits_dep, logits_word


def main(train_cfg='config/pretrain.json',
         model_cfg='config/bert_base.json',
         data_file='/root/voucher/dataset/tifu/bert/train.tsv',
         model_file=None,
         data_parallel=True,
         word_vocab='/root/voucher/dataset/tifu/bert/word_vocab.txt',
         pos_vocab='/root/voucher/dataset/tifu/bert/pos_vocab.txt',
         dep_vocab='/root/voucher/dataset/tifu/bert/dep_vocab.txt',
         pos_dep_word_vocab='/root/voucher/dataset/tifu/bert/pos_dep_word.pkl',
         save_dir='../exp/bert/pretrain',
         log_dir='../exp/bert/pretrain/runs',
         max_len=384,
         max_pred=20,
         mask_prob=0.15,
         mode=train):

    if mode == 'train':
        pass
    elif mode == 'eval':
        pass
    #    max_pred = max_len
    #    mask_prob = 1
    else:
        print("please select correct mode")
        exit(1)

    cfg = train.Config.from_json(train_cfg)
    model_cfg = models.Config.from_json(model_cfg)

    set_seeds(cfg.seed)

    custom_tokenizer = CustomVocabTokenizer(word_vocab_file=word_vocab,
                                            pos_vocab_file=pos_vocab,
                                            dep_vocab_file=dep_vocab,
                                            pos_dep_word_vocab_file=pos_dep_word_vocab)
    custom_tokenize = lambda word, pos, dep: custom_tokenizer.tokenize(custom_tokenizer.convert_to_unicode(word),
                                                                       custom_tokenizer.convert_to_unicode(pos),
                                                                       custom_tokenizer.convert_to_unicode(dep))

    pipeline = [Preprocess4Pretrain(max_pred,
                                    mask_prob,
                                    list(custom_tokenizer.word_tokenizer.vocab.keys()),
                                    list(custom_tokenizer.pos_tokenizer.vocab.keys()),
                                    list(custom_tokenizer.dep_tokenizer.vocab.keys()),
                                    custom_tokenizer.convert_tokens_to_ids,
                                    max_len)]
    data_iter = TifuDataLoader(data_file,
                               cfg.batch_size,
                               custom_tokenize,
                               max_len,
                               pipeline=pipeline)

    model = BertModel4Pretrain(model_cfg,
                               custom_tokenizer.get_word_vocab_size(),
                               custom_tokenizer.get_pos_vocab_size(),
                               custom_tokenizer.get_dep_vocab_size())
    criterion1 = nn.CrossEntropyLoss(reduction='none')
    criterion2 = nn.CrossEntropyLoss(reduction='none')
    criterion3 = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.optim4GPU(cfg, model)
    trainer = train.Trainer(cfg, model, data_iter, optimizer, save_dir, get_device())

    writer = SummaryWriter(log_dir=log_dir) # for tensorboardX

    if mode == 'train':
        def get_loss(model, batch, global_step): # make sure loss is tensor
            input_word_ids,\
            input_pos_ids,\
            input_dep_ids,\
            input_segment_ids,\
            input_mask,\
            masked_word_ids,\
            masked_pos,\
            masked_weights,\
            target_word_ids,\
            target_pos_ids,\
            target_dep_ids,\
            target_mask = batch

            logits_pos, logits_dep, logits_word = model(input_word_ids,
                                                        input_segment_ids,
                                                        masked_pos,
                                                        input_mask,
                                                        target_word_ids,
                                                        target_mask)

            loss_pos = criterion1(logits_pos.transpose(1, 2), input_pos_ids) # for masked pos
            loss_pos = (loss_pos*input_mask.float()).mean()

            loss_dep = criterion2(logits_dep.transpose(1, 2), input_dep_ids) # for masked dep
            loss_dep = (loss_dep*input_mask.float()).mean()

            loss_word = criterion3(logits_word.transpose(1, 2), masked_word_ids) # for masked word
            loss_word = (loss_word*masked_weights.float()).mean()
            print(loss_pos.item(), loss_dep.item(), loss_word.item())
            writer.add_scalars('data/scalar_group',
                               {'loss_pos': loss_pos.item(),
                                'loss_dep': loss_dep.item(),
                                'loss_word': loss_word.item(),
                                'loss_total': (loss_pos + loss_dep + loss_word).item(),
                                'lr': optimizer.get_lr()[0],
                               },
                               global_step)

            return loss_pos + loss_dep + loss_word

        trainer.train(get_loss, model_file, None, data_parallel)
    elif mode == 'eval':
        def evaluate(model, batch):
            input_word_ids,\
            input_pos_ids,\
            input_dep_ids,\
            input_segment_ids,\
            input_mask,\
            masked_word_ids,\
            masked_pos,\
            masked_weights,\
            target_word_ids,\
            target_pos_ids,\
            target_dep_ids,\
            target_mask = batch
            logits_pos, logits_dep, logits_word = model(input_word_ids,
                                                        input_segment_ids,
                                                        masked_pos,
                                                        input_mask,
                                                        target_word_ids,
                                                        target_mask)
            _, label_pos = logits_pos.max(-1)
            result_pos = (label_pos == input_pos_ids).float() #.cpu().numpy()
            pos_accuracy = result_pos.mean()

            _, label_dep = logits_dep.max(-1)
            result_dep = (label_dep == input_dep_ids).float()
            dep_accuracy = result_dep.mean()

            _, label_word = logits_word.max(-1)
            result_word = (label_word == masked_word_ids).float()
            word_accuracy = result_word.mean()

            accuracies = [pos_accuracy, dep_accuracy, word_accuracy]
            results = [result_pos, result_dep, result_word]
            return accuracies, results

        results = trainer.eval(evaluate, model_file, data_parallel, eval_kind_names=["PosTagging", "SyntaxParsing", "Word"])
        print(results)


if __name__ == '__main__':
    fire.Fire(main)
