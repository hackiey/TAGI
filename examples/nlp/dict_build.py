import glob
import tagi
import math
import time
import multiprocessing

from tqdm import tqdm, trange
from collections import Counter, defaultdict
from functools import reduce
from operator import mul
from pygtrie import Trie

from tagi.nlp.utils.preprocess import Preprocessor


class DictBuilding:
    def __init__(self, min_n=2, max_n=4, min_freq=2):
        self.min_n = min_n
        self.max_n = max_n
        self.min_freq = min_freq

        self.preprocessor = Preprocessor()

        self.ngrams_freq = {}
        self.ngrams_words = {i: set() for i in range(1, max_n+2)}

    def union_freq(self, freq_chunk):
        t1 = time.time()
        for key in freq_chunk.keys():
            self.ngrams_freq[key] = freq_chunk[key] + self.ngrams_freq.get(key, 0)
        t2 = time.time()
        print('union_freq', t2 - t1)

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

    def cal_left_right_entropy(self):
        left_right_entropy = {}
        for n in range(self.min_n, self.max_n+1):
            ngrams_entropy = {}
            target_ngrams = self.ngrams_words[n]
            parent_words = self.ngrams_words[n+1]
            left_neighbors = Trie()
            right_neighbors = Trie()

            for parent_word in tqdm(parent_words, desc='build neighbors'):
                right_neighbors[parent_word] = self.ngrams_freq[parent_word]
                left_neighbors[parent_word[1:]+parent_word[0]] = self.ngrams_freq[parent_word]

            for target_ngram in tqdm(target_ngrams, desc='target ngram'):
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

            left_right_entropy.update(ngrams_entropy)

        return left_right_entropy

    def cal_ngram_pmi(self):
        # pmi: point wise mutual information
        # ami: average mutual information
        mi = {}
        n1_total_count = sum([self.ngrams_freq[k] for k in self.ngrams_words[1]])
        for n in range(self.min_n, self.max_n+1):
            n_total_count = sum([self.ngrams_freq[k] for k in self.ngrams_words[n]])
            for target_ngram in tqdm(self.ngrams_words[n], desc='{} ngrams'.format(n)):
                target_ngram_freq = self.ngrams_freq[target_ngram]
                joint_prob = target_ngram_freq / n_total_count
                indep_prob = reduce(mul, [self.ngrams_freq[c] for c in target_ngram])/((n1_total_count)**n)
                pmi = math.log(joint_prob/indep_prob, 2)
                ami = pmi / len(target_ngram)

                mi[target_ngram] = (pmi, ami)

        return mi

    def build(self, left_right_entropy, mi):
        words = mi.keys()
        word_liberalization = lambda le, re: math.log((le * 2 ** re + re * 2 ** le+0.00001)/(abs(le - re)+1), 1.5)
        word_info_scores = {
            word: (mi[word][0], mi[word][1], left_right_entropy[word][0], left_right_entropy[word][1],
                   min(left_right_entropy[word][0], left_right_entropy[word][1]),
                   word_liberalization(left_right_entropy[word][0], left_right_entropy[word][1])+mi[word][1])
            for word in words}

        target_ngrams = word_info_scores.keys()
        start_chars = Counter([n[0] for n in target_ngrams])
        end_chars = Counter([n[-1] for n in target_ngrams])
        threshold = int(len(target_ngrams) * 0.004)
        threshold = max(50, threshold)

        invalid_start_chars = set([char for char, count in start_chars.items() if count > threshold])
        invalid_end_chars = set([char for char, count in end_chars.items() if count > threshold])
        invalid_target_ngrams = set([n for n in target_ngrams if (n[0] in invalid_start_chars or n[-1] in invalid_end_chars)])

        for n in invalid_target_ngrams:
            word_info_scores.pop(n)

        return word_info_scores


if __name__ == '__main__':
    import os
    import pickle

    dict_building = DictBuilding()
    preprocessor = Preprocessor()
    if not os.path.exists('tmp_left_right_entropy.pkl'):
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

        print('ngrams data load completed, start calculate left right entropy...')
        left_right_entropy = dict_building.cal_left_right_entropy()
        pickle.dump(left_right_entropy, open('tmp_left_right_entropy.pkl', 'wb'))
    else:
        left_right_entropy = pickle.load(open('tmp_left_right_entropy.pkl', 'rb'))
        print('left right entropy load completed')

    if not os.path.exists('tmp_mi.pkl'):
        print('start calculate ngram pmi and ami')
        mi = dict_building.cal_ngram_pmi()
        pickle.dump(mi, open('tmp_mi.pkl', 'wb'))
    else:
        mi = pickle.load(open('tmp_mi.pkl', 'rb'))
        print('mi load completed')

    word_info_scores = dict_building.build(left_right_entropy, mi)
    pickle.dump(word_info_scores, open('word_info_scores.pkl', 'wb'))
