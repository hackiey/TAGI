import re
import json
import multiprocessing
from tqdm import tqdm


class Preprocessor:
    def __init__(self):
        pass

    @staticmethod
    def sentence_split(texts, strict=True, lang='zh'):
        if lang == 'zh':
            regex_split_zh = u'[^\u4e00-\u9fa50-9a-zA-Z]+'
        else:
            regex_split_zh = ''

        if isinstance(texts, list):
            all_texts = []
            for text in texts:
                if strict:
                    all_texts.extend(re.split(regex_split_zh, text))
                else:
                    all_texts.append(text)
        else:
            all_texts = re.split(regex_split_zh, texts)

        return [text for text in all_texts if len(text) > 0]

    def word_count(self, texts, filter=False, lang='zh'):
        if filter is True:
            texts = self.sentence_split(texts, lang=lang)

        return sum([len(text) for text in texts])

    @staticmethod
    def ngram_generator(corpus, n):
        def _ngram_generator_str(text: str, n):
            for i in range(len(text)-n+1):
                yield text[i:i+n]

        if isinstance(corpus, str):
            for ngram in _ngram_generator_str(corpus, n):
                yield ngram
        elif isinstance(corpus, list):
            for text in corpus:
                for ngram in _ngram_generator_str(text, n):
                    yield ngram

    @staticmethod
    def read_corpus(file_path, dtype='jsonl', key='text'):
        if dtype == 'jsonl':
            with open(file_path, 'r') as f:
                corpus = []
                for line in f.readlines():
                    corpus.append(json.loads(line)[key])

        elif dtype == 'json':
            corpus = json.load(open(file_path, 'r')[key])
        else:
            corpus = []

        return corpus
