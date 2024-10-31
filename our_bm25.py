from typing import List, Dict, Optional, Union
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

class OurBM25:
  def __init__(self, k1=1.2, b=0.75, epsilon=0.75):
    self.k1 = k1
    self.b = b
    self.epsilon = epsilon
    self.corpus_id = None
    self.top_k = None

    #needs to be saved
    self.document_score = None
    self.word_index = dict()
    
    #no need to save
    self.sum_doc_len, self.corpus_size = 0, 0  #for avg_doc_len
    self.avg_doc_len = 0
    self.total_word_freq = defaultdict(int) #1 if word in doc else 0 #idf
    self.word2idf = dict() #idf
    self.d = 0

  def _filter(self, word) -> bool:
    return isinstance(word, str) and word.isprintable() and word != ""

  def _split(self, doc)-> List[str]:
    return list(filter(self._filter, doc.split(' ')))

  def _build_idf(self):
    idf_sum, idf_len = 0, 0
    word_negative_idfs = []

    for word, freq in self.total_word_freq.items():
      idf = np.log((self.corpus_size + 1) / (freq), dtype=np.float32)
      self.word2idf[word] = idf
      idf_len += 1
      idf_sum += idf
      if idf < 0:
        word_negative_idfs.append(word)

    average_idf = idf_sum / idf_len
    eps = self.epsilon * average_idf
    self.word2idf.update({word: eps for word in word_negative_idfs})
    
  def fit_transform(self, 
                    corpus : List[str],
                    top_k :int=None,
                    round_decimals : 2):
    #local variables for transforming corpus
    self.top_k = top_k
    doc_word_freq = []
    docs_splitted = []

    for doc in corpus:
      splitted_doc = self._split(doc)
      word_freq_doc = Counter(splitted_doc)

      if self.top_k:
        word_freq_doc = dict(word_freq_doc.most_common(self.top_k))
        top_k_filter = lambda x : x in word_freq_doc
        splitted_doc = list(filter(top_k_filter, splitted_doc))

      #store doc len
      self.sum_doc_len += len(splitted_doc)
      self.corpus_size += 1

      #store word_freq per doc
      for unique_word, _ in word_freq_doc.items():
        self.total_word_freq[unique_word] += 1

      #save for transforming
      doc_word_freq.append(word_freq_doc)
      docs_splitted.append(splitted_doc)
    
    self.avg_doc_len = self.sum_doc_len / self.corpus_size
    self._build_idf() #update word2idf
    self.word_index = {word: i for i, word in enumerate(self.word2idf.keys())}
    self.d = len(self.word2idf.keys())
    #transform and save numpy array
    self._transform(docs_splitted, doc_word_freq, round_decimals)
    
  def _transform(self, 
                 docs_splitted : List[List[str]],
                 doc_word_freq : List[Dict[str, int]],
                 round_decimals):    
    score_vecs = np.zeros([self.d, self.corpus_size], dtype=np.float32)
    for doc_idx, doc in enumerate(tqdm(docs_splitted)):
      doc_freqs = doc_word_freq[doc_idx]

      for word in doc_freqs.keys(): #use as word set
        x = self.word2idf[word] * doc_freqs[word] * (self.k1 + 1)
        y = doc_freqs[word] + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_doc_len)
        score = (x / y) + 1 #todo
        score_vecs[self.word_index[word], doc_idx] = np.round(score, round_decimals)

    self.document_score = score_vecs

  def search(self, 
             query : Union[List[str], str], 
             k=10, 
             return_score:bool=True):
    splitted_query = self._split(query) if type(query) == str else query
    word_indices = [self.word_index[w] for w in splitted_query if w in self.word_index] # num_word * num_docs 
    scores = sum(self.document_score[word_indices, :]) # 1 * num_docs
    top_doc_k = np.argsort(scores)[::-1][:k]
    if return_score == False:
      return self.corpus_id[top_doc_k] if self.corpus_id else top_doc_k
    else:
      return [(scores[idx], idx) for idx in top_doc_k]
