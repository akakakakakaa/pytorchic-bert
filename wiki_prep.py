import fire
import glob
import re
import random
import numpy as np
import csv

#spm_train --input=all_line.txt --model_prefix=sentencepiece --vocab_size=64000 --character_coverage=1.0 --model_type=unigram --input_sentence_size=1000000 --shuffle_input_sentence=true
def split_wiki_text_to_paragraphs(wiki_path, output_csv_path, type):
    if type!="classify" and type!="summarize":
        print("please use type classify or summarize")

    data_lists = []

    file_list = list(glob.iglob(wiki_path+'/**/wiki_*', recursive=True))
    file_num = len(file_list)

    processed = 0
    txt_f = open(wiki_path+"/all_line.txt", 'w')
    for filename in file_list:
        with open(filename, 'r') as f:
            content = f.read()
            sentences = re.split('<doc>\n|</doc>\n|\n', content)
            sentences = list(filter(None, sentences))

            target = ''
            input = ''
            data_list = []
            lines= []
            for sentence in sentences:
                if sentence.find("==") != -1:
                    if input:
                        data_list.append({"target": target, "input": input[:-1]})
                        input = ''

                    if sentence.find("title") != -1:
                        #title==
                        target = sentence[7:]
                        if len(data_list) > 0:
                            data_lists.append(data_list)
                            data_list = []

                        if len(lines) > 1:
                            for line in lines:
                                txt_f.write(line)
                            txt_f.write('\n')
                        lines = []
                    else:
                        #section${number}==
                        target = sentence[10:]
                else:
                    for txt_str in re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", sentence):
                        #txt_f.write(re.sub("[\(\[].*?[\)\]]", "",str) + '\n')
                        if len(txt_str) > 0:
                            lines.append(txt_str + '\n')
                    input += sentence + ' '

        processed += 1
        print("%s %d/%d processing..." % (filename, processed, file_num))
    print("DONE preprocessing")

    print("create csv")
    if type=="summarize":
        summarization_list = []
        for data_list in data_lists:
            for data in data_list:
                summarization_list.append(data)

        random.shuffle(summarization_list)
        keys = summarization_list[0].keys()

        print("num_data: " + str(len(summarization_list)))
        train_len = int(len(summarization_list) * 0.72)
        test_len = int(len(summarization_list) * 0.18)
        print("train_len: " + str(train_len))
        print("test_len: " + str(test_len))

        with open(output_csv_path+"/train.csv", 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(summarization_list[:train_len])

        with open(output_csv_path+"/test.csv", 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(summarization_list[train_len:train_len+test_len])

        with open(output_csv_path+"/eval.csv", 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(summarization_list[train_len+test_len:])

    elif type=="classify":
        not_same_list = []
        same_list_for_csv = []
        not_same_list_for_csv = []
        for data_list in data_lists:
            input = [data["input"] for data in data_list]
            random.shuffle(input)

            split_num = len(input) // 2
            if split_num % 2 == 1:
                split_num += random.choices([-1, 1], [0.25, 0.75])[0]
                is_same = input[:split_num]
                not_same = input[split_num:]
            else:
                is_same = input[:split_num]
                not_same = input[split_num:]

            read_len = 0
            while read_len < len(is_same):
                same_list_for_csv.append({"label":"0", "a":is_same[read_len], "b":is_same[read_len+1]})
                read_len += 2

            if len(not_same) > 0:
                not_same_list.append(not_same)

        #짝을 매칭시켜주다 리스트가 남을 수 있기 때문에, 1 이상으로 설정해줍니다.
        while len(not_same_list) > 2:
            if len(not_same_list) % 10000 == 0:
                print(str(len(not_same_list)) + " remained")

            #someone says random is 15x faster than randint
            #choose_left = random.randint(0, len(not_same_list)-2)
            #choose_right = random.randint(choose_left+1, len(not_same_list)-1)
            choose_left = int(random.random() * len(not_same_list))
            while True:
                choose_right = int(random.random() * len(not_same_list))
                if choose_left != choose_right:
                    break

            choose_left_idx = int(random.random() * len(not_same_list[choose_left]))
            choose_right_idx = int(random.random() * len(not_same_list[choose_right]))

            not_same_list_for_csv.append({"label":"1", "a":not_same_list[choose_left][choose_left_idx], "b":not_same_list[choose_right][choose_right_idx]})

            del not_same_list[choose_left][choose_left_idx]
            del not_same_list[choose_right][choose_right_idx]

            remain_left_len = len(not_same_list[choose_left])
            remain_right_len = len(not_same_list[choose_right])

            if remain_left_len == 0 and remain_right_len == 0:
                if choose_right > choose_left:
                    del not_same_list[choose_right]
                    del not_same_list[choose_left]
                else:
                    del not_same_list[choose_left]
                    del not_same_list[choose_right]
            elif remain_left_len == 0:
                del not_same_list[choose_left]
            elif remain_right_len == 0:
                del not_same_list[choose_right]

        result_list = same_list_for_csv + not_same_list_for_csv
        random.shuffle(result_list)
        keys = result_list[0].keys()

        print("num_data: " + str(len(result_list)))
        train_len = 320000#int(len(result_list) * 0.72)
        test_len = 64000#int(len(result_list) * 0.18)
        print("train_len: " + str(train_len))
        print("test_len: " + str(test_len))
        with open(output_csv_path+"/train.tsv", 'w') as f:
            dict_writer = csv.DictWriter(f, keys, delimiter="\t")
            dict_writer.writeheader()
            dict_writer.writerows(result_list[:train_len])

        with open(output_csv_path+"/dev.tsv", 'w') as f:
            dict_writer = csv.DictWriter(f, keys, delimiter="\t")
            dict_writer.writeheader()
            dict_writer.writerows(result_list[train_len:train_len+test_len])

        with open(output_csv_path+"/test.tsv", 'w') as f:
            dict_writer = csv.DictWriter(f, keys, delimiter="\t")
            dict_writer.writeheader()
            dict_writer.writerows(result_list[train_len+test_len:])

    print("FINISH")

def main(wiki_path='../wiki/bakgua',
         output_csv_path='./prep/wiki/bakgua',
         type='classify'):
    split_wiki_text_to_paragraphs(wiki_path, output_csv_path, type)


if __name__ == '__main__':
    fire.Fire(main)