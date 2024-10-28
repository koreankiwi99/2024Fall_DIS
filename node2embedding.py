from tqdm import tqdm
import gc
import torch
import argparse
import os
from utils import LoadNodes
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def chunk(lst, chunk_size):
  return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def main(args):
  embed_model = HuggingFaceEmbedding(model_name=args.model_name,
                                     embed_batch_size=args.batch_size,
                                     device=args.device)
  loaded_nodes = LoadNodes(args.save_node_dir).nodes
  splitted_corpus = [node.text for node in loaded_nodes]
  os.mkdir(args.save_embedding_dir)

  if args.sharded:
    progress_bar = tqdm(range(len(splitted_corpus)))
    chunked_corpus = chunk(splitted_corpus, args.num_sharded_doc)

    for idx, chuncked_node in enumerate(chunked_corpus):
      embeddings = embed_model.get_text_embedding_batch(chuncked_node, show_progress=True)
      tmp_embed = torch.zeros([args.num_sharded_doc, args.d])

      for e_idx, embed in enumerate(embeddings):
        embed = torch.tensor(embed)
        tmp_embed[e_idx,:args.d] = embed

      torch.save(tmp_embed, os.path.join(args.save_embedding_dir,f'{idx}.pt'))
      del tmp_embed, embeddings
      torch.cuda.empty_cache()
      gc.collect()
      progress_bar.update(len(chuncked_node))

  else:
    embeddings = embed_model.get_text_embedding_batch(splitted_corpus,
                                                      show_progress=True)
    torch.save(torch.tensor(embeddings), os.path.join(args.save_embedding_dir,'total.pt'))

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--sharded', type=bool, default=True)
  parser.add_argument('--save_node_dir', type=str, required=True)
  parser.add_argument('--save_embedding_dir', type=str, required=True)
  parser.add_argument('--num_sharded_doc', type=int, default=10240)
  parser.add_argument('--d', type=int, default=384)
  parser.add_argument('--batch_size', type=int, default=2048)
  parser.add_argument('--model_name', type=str, required=True)
  parser.add_argument('--device', type=str, default='cuda')
  return parser.parse_args()

if __name__ == '__main__':
  args = parse_arguments()
  main(args)
