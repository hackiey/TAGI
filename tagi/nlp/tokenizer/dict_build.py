import re
import numpy as np

from collections import defaultdict
from tqdm import tqdm

class DictBuilding:
    def __init__(self, min_count=10, min_pmi=0):
        self.min_count = min_count
        self.min_pmi = min_pmi
        self.chars, self.pairs = defaultdict(int), defaultdict(int)

        self.total = 0

    def text_filter(self, texts):
        for text in texts:
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', text):
                if t:
                    yield t

    def count(self, texts):
        for text in tqdm(self.text_filter(texts), desc=''):
            self.chars[text[0]] += 1

            for i in range(len(text) - 1):
                self.chars[text[i+1]] += 1
                self.pairs[text[i:i+2]] += 1

                self.total += 1

        self.chars = {char: count for char, count in self.chars.items() if count > self.min_count}
        self.pairs = {pair: count for pair, count in self.pairs.items() if count > self.min_count}

        self.strong_segments = set()
        for pair, count in self.pairs.items():
            _ = np.log(self.total * count / (self.chars[pair[0]] * self.chars[pair[1]]))
            if _ >= self.min_pmi:
                self.strong_segments.add(pair)

    def build_dict(self, texts):
        self.words = defaultdict(int)

        for text in self.text_filter(texts):
            s = text[0]
            for i in range(len(text)-1):
                if text[i:i+2] in self.strong_segments: #如果比较“密切”则不断开
                    s += text[i+1]
                else:
                    self.words[s] += 1 #否则断开，前述片段作为一个词来统计
                    s = text[i+1]
            self.words[s] += 1 #最后一个“词”

        self.words = {word: count for word, count in self.words.items() if count >= self.min_count} #最后再次根据频数过滤




# file_paths = glob.glob('../../data/corpus/NLPIR-news-corpus/*.txt')
#
# texts = []
#
# for file_path in file_paths:
#     with open(file_path, 'r', encoding="utf8") as f:
#         for line in f.readlines():
#             texts.append(line)
#
# dict_building = DictBuilding(16, 1)
# dict_building.count(texts)
# dict_building.build_dict(texts)
#
# print(1)