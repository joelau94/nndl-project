import sys
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict, Counter
import cPickle as pkl

def w2i_mapping(input_file, output_file, min_freq=5):
	sents = map(word_tokenize, sent_tokenize(open(input_file,'r').read().decode('utf-8').lower()))
	i2w = ['_UNK_'] + [word for word, freq in Counter(reduce(lambda x,y: x+y, sents)).iteritems() if freq >= min_freq]
	print('Total: {} sentences, {} unique tokens.'.format(len(sents), len(i2w)))
	w2i = defaultdict(int)
	w2i.update({w:i for i,w in enumerate(i2w)})
	ids = map(lambda l: map(lambda w: str(w2i[w]), l), sents)
	pkl.dump(i2w, open('i2w.pkl','w+'))
	pkl.dump(w2i, open('w2i.pkl','w+'))
	open(output_file,'w+').write('\n'.join(map(lambda l: ' '.join(l), ids)).encode('utf-8'))

def main():
	w2i_mapping(input_file=sys.argv[1], output_file=sys.argv[2], min_freq=5)

if __name__ == '__main__':
	main()