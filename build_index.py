from utils import LoadNodes
import os
import torch
from tqdm import tqdm
import argparse

import faiss
import faiss.contrib.torch_utils
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def main(args):
  loaded_nodes = LoadNodes(args.save_node_dir).nodes
  files = [os.path.join(args.save_embedding_dir, f) for f in os.listdir(args.save_embedding_dir)]
  files = sorted(files, key = lambda x : int(x.split('/')[-1].split('.')[0]))
  embeddings = torch.concat([torch.load(_) for _ in files])[:len(loaded_nodes),:].numpy() #todo

  for node, embedding in zip(tqdm(loaded_nodes), embeddings):
    node.embedding = embedding#.tolist()
    #node.text = '' #not efficient

  embed_model = HuggingFaceEmbedding(model_name=args.model_name)
  Settings.embed_model = embed_model

  if args.index_type == 'flatL2':
    faiss_index = faiss.IndexFlatL2(args.d)
  
  elif args.index_type == 'ivfflat':
    quantizer = faiss.IndexFlatL2(args.d)
    faiss_index = faiss.IndexIVFFlat(quantizer, args.d, args.n_list)
    #gpu_resource = faiss.StandardGpuResources()
    #gpu_quantizer = faiss.index_cpu_to_gpu(gpu_resource, 0, quantizer)
    #faiss_index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
    faiss_index.train(embeddings)
  
  elif args.index_type == 'ivfpq':#todo
    #faiss_index = faiss.IndexIVFPQ(quantizer, args.d, args.n_list, m, args.nbits)
    #faiss_index.train(embeddings)
    pass

  vector_store = FaissVectorStore(faiss_index=faiss_index)
  storage_context = StorageContext.from_defaults(vector_store=vector_store)
  index = VectorStoreIndex(loaded_nodes, storage_context=storage_context)
  os.mkdir(args.save_index_dir)
  index.storage_context.persist(persist_dir=args.save_index_dir)

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--save_node_dir', type=str, required=True)
  parser.add_argument('--save_embedding_dir', type=str, required=True)
  parser.add_argument('--n_list', type=int, default=100)
  parser.add_argument('--d', type=int, default=384)
  parser.add_argument('--index_type', type=str, default='ivfflat')
  parser.add_argument('--save_index_dir', type=str, required=True)
  parser.add_argument('--model_name', type=str, required=True)
  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_arguments()
  main(args)
