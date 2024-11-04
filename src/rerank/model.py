from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict
import numpy as np

class ReRanker:
  def __init__(self, 
               model_name : str, 
               device : str,
               id2doc = Dict[str,str]):
    self.device = device
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                    device_map = self.device)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name,  
                                                   device_map = self.device)
    self.id2doc = id2doc
    self.tokenizer.model_max_length = 512

  def get_score(self, 
                sentence_pairs: List[List[str]], 
                k : int=10):
    """
    Get score for each sentence pair
    """
    features = self.tokenizer(sentence_pairs,
                              padding=True,
                              truncation=True,
                              return_tensors="pt").to(self.device)
    with torch.no_grad():
      similarity_scores = self.model(**features).logits
  
    return similarity_scores[:, 0].cpu().numpy()
  
  def rerank(self, 
             query: List[str], 
             doc_ids: List[List[str]], 
             k : int=10, 
             return_score:bool = False):
    """
    Retrieve documents for each query
    """
    sentence_pairs = [[query, self.id2doc[doc_id]] for doc_id in doc_ids]
    similarity_scores = self.get_score(sentence_pairs, k)
    if return_score == True:
      return similarity_scores

    else:
      top_k_doc = np.argsort(similarity_scores)[::-1][:k]
      return [doc_ids[idx] for idx in top_k_doc]


class HybridSearch:
  def __init__(self, 
               bm_scores: List[np.array], 
               rerank_scores : List[np.array], 
               doc_ids : List[List[str]]):
    """
    Hybrid search using BM25 and Reranker scores
    """
    self.bm_scores = bm_scores
    self.rerank_scores = rerank_scores
    self.doc_ids = doc_ids
    self.output = self._build()

  def normalize(self, scores : np.array):
    """
    Normalize scores to [0,1]
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    return (scores - min_score) / (max_score - min_score)

  def hybrid_search(self, 
                    bm : np.array, 
                    rerank : np.array, 
                    doc_ids : List[str]):
    """
    Hybrid search using BM25 and Reranker scores
    """
    score = self.normalize(bm) + self.normalize(rerank)
    top_doc = np.argsort(score)[::-1][:10]
    return [doc_ids[idx] for idx in top_doc]

  def _build(self):
    """
    Build output
    """
    return [self.hybrid_search(b, r, i) for b,r,i in zip(self.bm_scores,
                                                         self.rerank_scores,
                                                         self.doc_ids)]
