import itertools
import numpy as np
import networkx as nx
import vocab

def coref_score(instance, property_id):
  return [ instance.subject_entity["coref_score"], instance.object_entity["coref_score"] ]

def el_score(instance, property_id):
  return [ instance.subject_entity["el_score"], instance.object_entity["el_score"] ]

def _entity_linker_types_from_mention(entity):
  arr = np.zeros(len(vocab.types), np.float32)
  for i, t in enumerate(vocab.types):
    if t in entity["types"]:
      arr[i] = 1.0
  return arr

def entity_linker_types(instance, property_id):
  return np.concatenate([
    _entity_linker_types_from_mention(instance.subject_entity),
    _entity_linker_types_from_mention(instance.object_entity)
  ])
def wikidata_predicates(instance, property_id):
  return None

def text_score(instance, property_id):
  return [ instance.text_instance.scores[property_id] ]