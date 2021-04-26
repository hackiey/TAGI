import glob
import tagi
import math
import time

from tqdm import tqdm, trange
from collections import Counter, defaultdict
from pygtrie import Trie

from tagi.nlp.utils.preprocess import Preprocessor


class DictBuilding:
    def __init__(self, min_n=2, max_n=4, min_freq=2):
        self.min_n = min_n
        self.max_n = max_n
        self.min_freq = min_freq

        self.preprocessor = Preprocessor()

    def union_freq(self, freq1, freq2):
        for key in freq1.keys():
            freq2[key] = freq1.get(key, 0) + freq2.get(key, 0)
        # words = freq1.keys() | freq2.keys()
        # freq = {}
        # for key in words:
        #     freq[key] = freq1.get(key, 0) + freq2.get(key, 0)

    @staticmethod
    def build_chunk(chunk, min_n, max_n, min_freq):
        ngrams_chunk = {}
        ngrams_words = {i: set() for i in range(1, max_n + 2)}
        for n in [1] + list(range(min_n, max_n+2)):
            n_generator = Preprocessor.ngram_generator(chunk, n)
            chunk_freq = dict(Counter(n_generator))
            ngrams_words[n].update(chunk_freq.keys())
            ngrams_chunk.update(chunk_freq)

        ngrams_chunk = {word: count for word, count in ngrams_chunk.items() if count >= min_freq}
        return ngrams_chunk, ngrams_words

    def build_ngrams(self, corpus, chunk_size=1000):

        ngrams_words = {i: set() for i in range(1, self.max_n + 2)}
        ngrams_freq = defaultdict(int)

        # def _build_chunk(chunk):
        #     ngrams_chunk = {}
        #     for n in [1] + list(range(self.min_n, self.max_n+2)):
        #         n_generator = self.preprocessor.ngram_generator(chunk, n)
        #         chunk_freq = dict(Counter(n_generator))
        #         ngrams_words[n].update(chunk_freq.keys())
        #         ngrams_chunk.update(chunk_freq)
        #
        #     ngrams_chunk = {word: count for word, count in ngrams_chunk.items() if count >= self.min_freq}
        #     return ngrams_chunk

        for i in trange(0, len(corpus), chunk_size, desc='chunk'):
            ngrams_chunk = _build_chunk(corpus[i:i+chunk_size])
            self.union_freq(ngrams_chunk, ngrams_freq)

        return ngrams_words, ngrams_freq

    def cal_ngram_entropy(self, parent_word_freq):
        total_count = sum(parent_word_freq)
        parent_word_probs = map(lambda x: x/total_count, parent_word_freq)
        entropy = sum(map(lambda x: -1 * x * math.log(x, 2), parent_word_probs))
        return entropy

    def cal_left_right_entropy(self, ngrams_words, ngrams_freq):
        entropy = {}
        for n in range(self.min_n, self.max_n):
            ngrams_entropy = {}
            target_ngrams = ngrams_words[n]
            parent_words = ngrams_words[n+1]
            left_neighbors = Trie()
            right_neighbors = Trie()

            for parent_word in parent_words:
                right_neighbors[parent_word] = ngrams_freq[parent_word]
                left_neighbors[parent_word[1:]+parent_word[0]] = ngrams_freq[parent_word]

            for target_ngram in target_ngrams:
                try:
                    right_neighbors_counts = (right_neighbors.values(target_ngram))
                    right_entropy = self.cal_ngram_entropy(right_neighbors_counts)
                except KeyError:
                    right_entropy = 0
                try:
                    left_neighbors_counts = (left_neighbors.values(target_ngram))
                    left_entropy = self.cal_ngram_entropy(left_neighbors_counts)
                except KeyError:
                    left_entropy = 0
                ngrams_entropy[target_ngram] = (left_entropy, right_entropy)

            entropy.update(ngrams_entropy)

        return entropy

    def build(self, corpus, chunk_size=1000):
        ngrams_words, ngrams_freq = self.build_ngrams(corpus, chunk_size)
        left_right_entropy = self.cal_left_right_entropy(ngrams_words, ngrams_freq)



if __name__ == '__main__':
    import os
    import json
    import pickle
    import multiprocessing

    dict_building = DictBuilding()
    preprocessor = Preprocessor()

    corpus_path = 'tmp_corpus.pkl'
    if os.path.exists(corpus_path):
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        file_paths = glob.glob('../../tagi/data/corpus/wiki_zh/*/*')

        processes_num = 16
        pool = multiprocessing.Pool(processes=processes_num)

        file_corpus = pool.map(Preprocessor.read_corpus, file_paths)
        corpus = []
        for c in file_corpus:
            corpus.extend(c)
        file_corpus = pool.map(Preprocessor.sentence_split, corpus)

        corpus = []
        for c in file_corpus:
            corpus.extend(c)
        pickle.dump(corpus, open('tmp_corpus.pkl', 'wb'))

    dict_building.build(corpus)
