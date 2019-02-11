import fire
import tensorflow as tf
from tensor2tensor.data_generators import text_encoder
import json
import spacy
from spacy.pipeline import Tagger
import csv
import operator
import pickle
import re

def tifu_prep(tifu_path, output_csv_path):
    nlp = spacy.load('en')


    datasets = []
    word_vocab = {}
    pos_vocab = {}
    dep_vocab = {}
    pos_dep_word_vocab = {}
    with open("../english-words/words.txt") as f:
        valid_words = set(f.read().split())

    for line in tf.gfile.Open(tifu_path, "rb"):
        line = text_encoder.to_unicode_utf8(line.strip())
        line_json = json.loads(line)
        if not line_json["tldr"]:
            continue

        inputs = line_json["selftext_without_tldr"].lstrip()
        targets = line_json["tldr"].lstrip()
        inputs_token = nlp(inputs)
        targets_token = nlp(targets)

        input_tokens = ""
        input_pos = ""
        input_syntax = ""
        for token in inputs_token:
            #while token.head != token:
            token_str = str(token)
            if token.pos_ == "SPACE":
                continue
            #if not token_str or ' ' in token_str or '\n' in token_str or u'\xa0' in token_str or '\r' in token_str:
            #    print(token.pos_)
            #    continue

            input_tokens += token_str + " "
            input_pos += token.pos_ + " "
            input_syntax += token.dep_ + " "

            if len(input_tokens.split()) != len(input_pos.split()) or len(input_tokens.split()) != len(input_syntax.split()):
                print(token_str, ord(token_str), ord(' '), token.pos_)

            if token_str in word_vocab.keys():
                word_vocab[token_str] += 1
            else:
                word_vocab[token_str] = 1

            if token.pos_ in pos_vocab.keys():
                pos_vocab[token.pos_] += 1
            else:
                pos_vocab[token.pos_] = 1

            if token.dep_ in dep_vocab.keys():
                dep_vocab[token.dep_] += 1
            else:
                dep_vocab[token.dep_] = 1

            if token.pos_ in pos_dep_word_vocab.keys():
                if token.dep_ in pos_dep_word_vocab[token.pos_].keys():
                    if not token_str in pos_dep_word_vocab[token.pos_][token.dep_]:
                        pos_dep_word_vocab[token.pos_][token.dep_].append(token_str)
                else:
                    pos_dep_word_vocab[token.pos_][token.dep_] = []
                    pos_dep_word_vocab[token.pos_][token.dep_].append(token_str)
            else:
                pos_dep_word_vocab[token.pos_] = {}
                pos_dep_word_vocab[token.pos_][token.dep_] = []
                pos_dep_word_vocab[token.pos_][token.dep_].append(token_str)

        target_tokens = ""
        target_pos = ""
        target_syntax = ""
        for token in targets_token:
            #while token.head != token:
            token_str = str(token)
            if token.pos_ == "SPACE":
                continue
            #if not token_str or ' ' in token_str or '\n' in token_str or u'\xa0' in token_str or '\r' in token_str:
            #    print(token.pos_)
            #    continue

            target_tokens += token_str + " "
            target_pos += token.pos_ + " "
            target_syntax += token.dep_ + " "
            if len(target_tokens.split()) != len(target_pos.split()) or len(target_tokens.split()) != len(target_syntax.split()):
                print(token_str, ord(token_str), ord(' '), token.pos_)

            if token_str in word_vocab.keys():
                word_vocab[token_str] += 1
            else:
                word_vocab[token_str] = 1

            if token.pos_ in pos_vocab.keys():
                pos_vocab[token.pos_] += 1
            else:
                pos_vocab[token.pos_] = 1

            if token.dep_ in dep_vocab.keys():
                dep_vocab[token.dep_] += 1
            else:
                dep_vocab[token.dep_] = 1

            if token.pos_ in pos_dep_word_vocab.keys():
                if token.dep_ in pos_dep_word_vocab[token.pos_].keys():
                    if not token_str in pos_dep_word_vocab[token.pos_][token.dep_]:
                        pos_dep_word_vocab[token.pos_][token.dep_].append(token_str)
                else:
                    pos_dep_word_vocab[token.pos_][token.dep_] = []
                    pos_dep_word_vocab[token.pos_][token.dep_].append(token_str)
            else:
                pos_dep_word_vocab[token.pos_] = {}
                pos_dep_word_vocab[token.pos_][token.dep_] = []
                pos_dep_word_vocab[token.pos_][token.dep_].append(token_str)


            #    token = token.head
        datasets.append({"input_token": input_tokens.rstrip(),
                         "input_pos": input_pos.rstrip(),
                         "input_syntax": input_syntax.rstrip(),
                         "target_token": target_tokens.rstrip(),
                         "target_pos": target_pos.rstrip(),
                         "target_syntax": target_syntax.rstrip()})

    print("num_data: " + str(len(datasets)))
    train_len = int(len(datasets) * 0.8)
    test_len = int(len(datasets) * 0.1)
    print("train_len: " + str(train_len))
    print("test_len: " + str(test_len))

    keys = datasets[0].keys()
    with open(output_csv_path+"/train.tsv", 'w') as f:
        dict_writer = csv.DictWriter(f, keys, delimiter="\t")
        dict_writer.writeheader()
        dict_writer.writerows(datasets[:train_len])

    with open(output_csv_path+"/dev.tsv", 'w') as f:
        dict_writer = csv.DictWriter(f, keys, delimiter="\t")
        dict_writer.writeheader()
        dict_writer.writerows(datasets[train_len:train_len+test_len])

    with open(output_csv_path+"/test.tsv", 'w') as f:
        dict_writer = csv.DictWriter(f, keys, delimiter="\t")
        dict_writer.writeheader()
        dict_writer.writerows(datasets[train_len+test_len:])

    with open(output_csv_path+"/pos_dep_word.pkl", 'wb') as f:
        pickle.dump(pos_dep_word_vocab, f)

    #sorted는 dictionary를 list로 바꿈
    sorted_word_vocab = sorted(word_vocab.items(), key=operator.itemgetter(1), reverse=True)
    with open(output_csv_path+"/word_vocab.txt", 'w') as f:
        for key, value in sorted_word_vocab:
            f.write(key + " " + str(value) + "\n")

    sorted_pos_vocab = sorted(pos_vocab.items(), key=operator.itemgetter(1), reverse=True)
    with open(output_csv_path+"/pos_vocab.txt", 'w') as f:
        for key, value in sorted_pos_vocab:
            f.write(key + " " + str(value) + "\n")

    sorted_dep_vocab = sorted(dep_vocab.items(), key=operator.itemgetter(1), reverse=True)
    with open(output_csv_path+"/dep_vocab.txt", 'w') as f:
        for key, value in sorted_dep_vocab:
            f.write(key + " " + str(value) + "\n")

    print("FINISH")


def main(tifu_path='../dataset/tifu/tifu_all_tokenized_and_filtered.json',
         output_csv_path='../dataset/tifu/bert'):
    tifu_prep(tifu_path, output_csv_path)


if __name__ == '__main__':
    fire.Fire(main)