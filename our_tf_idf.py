import numpy as np
from typing import List, Union
from tqdm import tqdm
from collections import defaultdict, Counter

class TF_IDF:
  def __init__(self):
    self.word_set = set()
    self.d = 0 #number of word_set
    self.word_index = {} #save word position in vectors
    self.total_n = 0
    self.top_k = None
    self.word_count = defaultdict(int)

  def _filter(self, word):
      return isinstance(word, str) and word.isprintable() and word != ""

  def _split(self, doc):
    return list(filter(self._filter, doc.split(' ')))

  def _count_words(self, docs):
    for doc in docs:
      for word in doc:
        self.word_count[word] = 1

  #def tf(self, doc, word):
  #  return sum([token == word for token in doc]) / len(doc)

  def idf(self, word):
    word_occurance = self.word_count.get(word, 0) + 1
    return np.log(self.total_n / word_occurance)

  def fit(self, corpus, return_matrix = True, top_k = 100):
    self.top_k = top_k
    self.total_n = len(corpus)

    docs_splitted = []
    for doc in corpus:
      splitted_doc = self._split(doc)
      if self.top_k:
        splitted_doc = [w for w, _ in Counter(splitted_doc).most_common(self.top_k)]

      self.word_set.update(splitted_doc)
      docs_splitted.append(splitted_doc)

    self.word_index = {word : idx for idx, word in enumerate(self.word_set)}
    self._count_words(docs_splitted)
    self.d = len(self.word_set)

    if return_matrix == True:
      return [self.tf_idf(doc) for doc in tqdm(docs_splitted)]

  def tf_idf(self, doc : Union[List, str]):
    vec = np.zeros([self.d,])
    doc = self._split(doc) if type(doc) == str else doc
    word_freq = Counter(doc)

    for word, count in word_freq.items():
      if word in self.word_index:
        tf = count / len(doc) #to shorten time
        idf = self.idf(word)
        vec[self.word_index[word]] = tf * idf

    return vec
