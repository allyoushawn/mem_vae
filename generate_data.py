import sys
from collections import defaultdict, Counter
from nltk import pos_tag
import pickle

data_dir=sys.argv[1]
with open(data_dir + '/train.txt', 'w') as op_f:
    with open('data/quora/train.csv') as f:
        for idx, line in enumerate(f.readlines()):
            if idx == 0: continue
            segments = line.rstrip().split(',')
            tokens1 = segments[4].split()
            tokens2 = segments[5].split()
            op_f.write(' '.join(tokens1[:30]) + '\t' + ' '.join(tokens2[:30]))
            op_f.write('\n')

with open(data_dir + '/test_ref.txt', 'w') as op_f:
    with open(data_dir + '/test_input.txt', 'w') as op_f2:
        with open('data/quora/test.csv') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0: continue
                segments = line.rstrip().split(',')
                tokens1 = segments[4].split()
                tokens2 = segments[5].split()
                op_f2.write(' '.join(tokens1[:30]) + '\t' + ' '.join(tokens2[:30]))
                op_f2.write('\n')
                op_f.write('{}'.format(' '.join(tokens2)))
                op_f.write('\n')

word2tag = defaultdict(Counter)
with open(data_dir + '/train.tag', 'w') as op_f:
    with open(data_dir + '/train.txt') as f:
        for line in f.readlines():
            line1, line2 = line.rstrip().split('\t')
            tags1 = [x[1] for x in pos_tag(line1.split())]
            tags2 = [x[1] for x in pos_tag(line2.split())]
            op_f.write(' '.join(tags1) + '\t' + ' '.join(tags2))
            op_f.write('\n')
            for i, x in enumerate(line1.split()):
                word2tag[x][tags1[i]] += 1
            for i, x in enumerate(line2.split()):
                word2tag[x][tags2[i]] += 1
with open(data_dir + '/word2tag.pkl', 'wb') as op_f:
    pickle.dump(word2tag, op_f)
