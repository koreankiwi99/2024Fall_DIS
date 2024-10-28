from utils import get_corpus, save_node, LoadNodes
from llama_index.core.node_parser import SentenceSplitter
import argparse
import os

def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--corpus_path', type=str, required=True)
  parser.add_argument('--corpus_path', type=int, default=1024)
  parser.add_argument('--save_node_dir', type=str, required=True)
  parser.add_argument('--sample', type=bool, default=False)

if __name__ == '__main__':
  args = parse_arguments()
  os.mkdir(args.save_node_dir)
  documents = get_corpus(args.corpus_path)
  if args.sample:
    documents = documents[:100]

  splitter = SentenceSplitter(args.chunk_size)
  nodes = splitter(documents, show_progress=True)
  save_node(args.save_node_dir, nodes)
