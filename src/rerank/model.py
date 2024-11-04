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
