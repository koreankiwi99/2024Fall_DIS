import numpy as np
from typing import List, Union, Optional, Dict
from tqdm import tqdm
import torch
from collections import defaultdict, Counter

class TfIdf:
  def __init__(self):
    self.word_set = set()
    self.d = 0 #number of word_set
    self.word_index = {} #save word position in vectors
    self.total_n = 0
    self.top_k = None
    self.word_count = defaultdict(int)

  def _filter(self, word) -> bool:
      return isinstance(word, str) and word.isprintable() and word != ""

  def _split(self, doc)-> List[str]:
    return list(filter(self._filter, doc.split(' ')))
     
  #def tf(self, doc, word):
  #  return sum([token == word for token in doc]) / len(doc)

  def idf(self, word: str) -> float:
    word_occurance = self.word_count.get(word, 0) + 1
    return np.log(self.total_n / word_occurance)

  def fit(self, 
          corpus : List[str], 
          return_matrix:bool = True, 
          top_k:int = 100) -> np.array:
    self.top_k = top_k
    self.total_n = len(corpus)

    #local variables : only to save time when transforming
    docs_splitted = []
    doc_word_freq = []

    for doc in corpus:
      splitted_doc = self._split(doc)
      word_freq_doc = Counter(splitted_doc)
      
      if self.top_k:
        word_freq_doc = dict(word_freq_doc.most_common(self.top_k))
        top_k_filter = lambda x : x in word_freq_doc
        splitted_doc = list(filter(top_k_filter, splitted_doc))

      for unique_word in word_freq_doc.keys():
        self.word_count[unique_word] += 1 # 1 if word in doc else 0 (for IDF)
        self.word_set.add(unique_word) #determine dimension

      doc_word_freq.append(word_freq_doc)
      docs_splitted.append(splitted_doc)

    self.word_index = {word : idx for idx, word in enumerate(self.word_set)}
    self.d = len(self.word_set)

    if return_matrix == True: #fit and transform
      return self.transform(docs_splitted, doc_word_freq)

  def transform(self, 
                doc_list : List[List[str]], 
                doc_word_freq : Optional[Dict[str, int]]=None) -> np.array:
    vec = np.zeros([len(doc_list),self.d], dtype=np.float32) # num of doc * dimension
    for doc_idx, doc in enumerate(tqdm(doc_list)):
      #if there is saved counter, use it
      word_freq = doc_word_freq[doc_idx] if doc_word_freq != None else Counter(doc)
      
      for word, count in word_freq.items():
        if word in self.word_index:
          tf = count / len(doc) #TF(t, d) d-current doc / query
          idf = self.idf(word) #IDF (t, D) D-whole corpus
          vec[doc_idx, self.word_index[word]] = tf * idf

    return vec

  def transform_query(self, doc : Union[List[str], str]) -> np.array:
    vec = np.zeros([self.d,], dtype=np.float32)
    doc = self._split(doc) if type(doc) == str else doc
    word_freq = Counter(doc)

    for word, count in word_freq.items():
      if word in self.word_index:
        tf = count / len(doc) #TF(t, d) d-current doc / query
        idf = self.idf(word) #IDF (t, D) D-whole corpus
        vec[self.word_index[word]] = tf * idf

    return vec
