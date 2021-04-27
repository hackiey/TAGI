import glob
import tagi
import math
import multiprocessing

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

        self.ngrams_freq = Trie()
        self.ngrams_words = {i: set() for i in range(1, max_n)}

    def union_freq(self, freq_chunk):
        for key in freq_chunk.keys():
            self.ngrams_freq[key] = freq_chunk[key] + self.ngrams_freq.get(key, 0)

    @staticmethod
    def build_chunk(chunk, min_n, max_n, min_freq, progress):
        print('build chunk', progress*100)
        freq_chunk = {}
        for n in [1] + list(range(min_n, max_n+2)):
            n_generator = Preprocessor.ngram_generator(chunk, n)
            chunk_freq = dict(Counter(n_generator))
            freq_chunk.update(chunk_freq)
        freq_chunk = {word: count for word, count in freq_chunk.items() if count >= min_freq}

        return freq_chunk

    def build_ngrams(self, corpus, chunk_size=10000, workers=16):
        pool = multiprocessing.Pool(processes=workers)
        for i in range(0, len(corpus), chunk_size):
            progress = i / len(corpus)
            pool.apply_async(DictBuilding.build_chunk,
                             args=(corpus[i:i+chunk_size], self.min_n, self.max_n, self.min_freq, progress),
                             callback=self.union_freq)
        pool.close()
        pool.join()

        for k in self.ngrams_freq.keys():
            self.ngrams_words[len(k)].add(k)

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

    def build(self):
        left_right_entropy = self.cal_left_right_entropy(self.ngrams_words, self.ngrams_freq)


if __name__ == '__main__':
    import os
    import pickle

    dict_building = DictBuilding()
    preprocessor = Preprocessor()

    if not (os.path.exists('tmp_words.pkl') and os.path.exists('tmp_freq.pkl')):
        if os.path.exists('tmp_corpus.pkl'):
            corpus = pickle.load(open('tmp_corpus.pkl', 'rb'))
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

        dict_building.build_ngrams(corpus, chunk_size=100000, workers=16)
        pickle.dump(dict_building.ngrams_freq, open('tmp_freq.pkl', 'wb'))
        pickle.dump(dict_building.ngrams_words, open('tmp_words.pkl', 'wb'))
    else:
        dict_building.ngrams_freq = pickle.load(open('tmp_freq.pkl', 'rb'))
        dict_building.ngrams_words = pickle.load(open('tmp_words.pkl', 'rb'))

    # dict_building.build()
