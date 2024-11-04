import json
import pickle
import pandas as pd
from ast import literal_eval
from typing import List
import gzip

from src.bm25.our_bm25 import OurBM25

def settings(config_path : str,
             embedding_path : str):
  """
  Load config file and embedding matrix
  """
  with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)

  bm25 = OurBM25()
  for k, v in config.items():
    setattr(bm25, k, v)
  
  if embedding_path.endswith('pkl'):
    with open(embedding_path, "rb") as f:
      matrix = pickle.load(f)
  
  elif embedding_path.endswith('npy.gz'):
    with gzip.GzipFile(embedding_path, "r") as f:
      matrix = np.load(f)

  bm25.document_score = matrix
  return bm25

def retrieval(bm25 : OurBM25, 
              queries : str, 
              top_k : int, 
              return_score:bool=True):
  """
  Retrieve documents for each query
  """
  return [bm25.search(q, top_k, return_score=return_score) for q in queries]

def check_recall(prediction : List[str], answers : List[str]):
  """
  Check recall of prediction
  """
  scores = [a in p for p, a in zip(prediction, answers)]
  return sum(scores) / len(scores)

def load_preprocessed(preprocessed_path :str, lang : str):
  """
  Load preprocessed data from csv file
  """
  df = pd.read_csv(preprocessed_path)
  lang_df = df[df['lang'] == lang].copy()
  lang_df['query'] = lang_df['query'].apply(lambda x: literal_eval(x))
  return list(lang_df['query']), list(lang_df['positive_docs'])
