import fire

def main(sentencepiece_path='../wiki/bakgua/sentencepiece.vocab',
         output_path='../wiki/bakgua/sentencepiece.txt'):
    in_f = open(sentencepiece_path, 'r')
    out_f = open(output_path, 'w')

    contents = in_f.readlines()
    contents = [x.strip() for x in contents]

    for content in contents:
        out_f.write(content.split('\t')[0] + '\n')

if __name__ == '__main__':
    fire.Fire(main)