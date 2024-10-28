import json
from typing import Dict, List, Union
from tqdm import tqdm
from functools import cached_property

from llama_index.core import Document
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
import pickle
import os

SOURCE = NodeRelationship.SOURCE
PREV = NodeRelationship.PREVIOUS
NEXT = NodeRelationship.NEXT

def wrap(doc: Dict[str, str]):
    return Document(text=doc['text'], metadata={"id": doc['docid'], "lang": doc['lang']})

def get_corpus(corpus_path):
  with open(corpus_path, 'r') as f:
    corpus = json.load(f)
  return list(map(wrap, tqdm(corpus)))

def extract_node(idx : int, node : TextNode):
  item = dict()
  item['id'] = node.id_
  item['text'] = node.text
  item['source'] = node.relationships[SOURCE].node_id
  item['metadata'] = node.metadata
  item['metadata']['num'] = idx
  item['prev'] = node.relationships[PREV].node_id if PREV in node.relationships else None
  item['next'] = node.relationships[NEXT].node_id if NEXT in node.relationships else None
  return item

def save_node(save_dir : str, nodes : List[TextNode]):
  path = os.path.join(save_dir, 'node.pkl')
  extracted_nodes = [extract_node(idx, node) for idx, node in enumerate(nodes)]

  with open(path, 'wb') as f:
    pickle.dump(extracted_nodes, f)

class LoadNodes:
  def __init__(self, node_dir : str) ->List[TextNode]:
    self.path = os.path.join(node_dir, 'node.pkl')
    self.loaded_nodes = self._load()
    self.nodes = self._build()

  def _load(self):
    with open(self.path, 'rb') as f:
      nodes = pickle.load(f)
    return nodes

  def _extract(self, item : Dict[str, Union[Dict[str,str], str]]):
    return TextNode(text=item['text'], id_ = item['id'], metadata = item['metadata'])

  def _build(self):
    leaf_nodes = {item['id'] : self._extract(item) for item in tqdm(self.loaded_nodes)}

    for item in tqdm(self.loaded_nodes):
      id = item['id']

      leaf_nodes[id].relationships[SOURCE] = RelatedNodeInfo(
          node_id=item['source'])

      if item['prev'] is not None:
        leaf_nodes[id].relationships[PREV] = RelatedNodeInfo(
            node_id=item['prev'])

      if item['next'] is not None:
        leaf_nodes[id].relationships[NEXT] = RelatedNodeInfo(
            node_id=item['next'])

    leaf_nodes = list(leaf_nodes.values())
    return sorted(leaf_nodes, key = lambda x : x.metadata['num'])
