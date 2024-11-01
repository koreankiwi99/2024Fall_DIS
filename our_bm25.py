from typing import List, Dict, Optional, Union
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from operator import itemgetter

class OurBM25:
  def __init__(self,
               k1=1.2,
               b=0.75,
               epsilon=0.75,
               corpus_id : Optional[List[str]]=None,
               top_k : Optional[int]=500,
               round_decimals : int=2,
               save_form : str='slow'):
    self.k1 = k1
    self.b = b
    self.epsilon = epsilon
    self.corpus_id = corpus_id
    self.top_k = top_k
    self.round_decimals = round_decimals
    self.save_form = save_form

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
                    corpus : List[str]):
    #local variables for transforming corpus
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
    self._transform(docs_splitted, doc_word_freq)

  def _transform(self,
                 docs_splitted : List[List[str]],
                 doc_word_freq : List[Dict[str, int]]):
    if self.save_form == 'fast':
      score_vecs = np.zeros([self.d, self.corpus_size], dtype=np.float32)

    elif self.save_form == 'economic':
      score_vecs = [dict() for _ in range(self.corpus_size)]

    elif self.save_form == 'slow':
      score_vecs = [defaultdict(int) for _ in range(self.corpus_size)]

    for doc_idx, doc in enumerate(tqdm(docs_splitted)):
      doc_freqs = doc_word_freq[doc_idx]

      for word in doc_freqs.keys(): #use as word set
        x = self.word2idf[word] * doc_freqs[word] * (self.k1 + 1)
        y = doc_freqs[word] + self.k1 * (1 - self.b + self.b * len(doc) / self.avg_doc_len)
        score = np.round((x / y) + 1, self.round_decimals) #todo

        if self.save_form == 'fast':#loading slow, fast retrieval
          score_vecs[self.word_index[word], doc_idx] = score

        elif self.save_form == 'economic':
          score_vecs[doc_idx][self.word_index[word]] = score

        elif self.save_form == 'slow':#loading fast, slow retrieval
          score_vecs[doc_idx][self.word_index[word]] = score

    self.document_score = score_vecs

  def search(self,
             query : Union[List[str], str],
             k : int=10,
             return_score : bool=True):
    splitted_query = self._split(query) if type(query) == str else query
    word_indices = [self.word_index[w] for w in splitted_query if w in self.word_index] # num_word * num_docs
    if len(word_indices) == 0:#not a single word in the query matching for word_set
      return []
               
    if self.save_form == 'fast': #loading slow, fast retrieval
      scores = sum(self.document_score[word_indices, :]) # 1 * num_docs

    elif self.save_form in ['slow', 'economic']:#loading fast, slow retrieval
      if len(word_indices) == 1:
        get_word_scores = lambda doc : doc[word_indices[0]]

      else:
        get_word_scores = lambda doc : sum(list(itemgetter(*word_indices)(doc)))

      scores = np.array([get_word_scores(doc) for doc in self.document_score])
    
    top_doc_k = np.argsort(scores)[::-1][:k]
    if return_score == False:
      return [self.corpus_id[idx] for idx in top_doc_k] if self.corpus_id else top_doc_k
      
    else:
      return [(scores[idx], idx) for idx in top_doc_k]
