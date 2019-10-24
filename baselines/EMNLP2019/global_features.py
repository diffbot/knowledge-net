import vocab
import numpy as np
import wikidata

def _entity_linker_types_from_mention(mention):
  candidates = mention._.uri_candidates
  types = [ t for cand in candidates for t in cand["types"] ]
  arr = np.zeros(len(vocab.types), np.float32)
  for i, t in enumerate(vocab.types):
    if t in types:
      arr[i] = 1.0
  return arr

def _global_properties_from_instance(wikidataProperties):
  '''
  single bit: How many properties link (any possible combination of) subject and object entity
  '''
  all_properties = set([p for l in wikidataProperties.values() for p in l if not p in ""])
  arr = np.zeros(1)
  arr[0] = len(all_properties)
  return arr

def _specific_properties_from_instance(wikidataProperties):
  '''
  one-hot vector: Which properties link (any possible combination of) subject and object entity
  '''
  all_properties = set([p for l in wikidataProperties.values() for p in l if not p in ""])
  arr = np.zeros(len(vocab.properties), np.float32)
  for i, t in enumerate(vocab.properties):
    if t in all_properties:
      arr[i] = 1.0
  return arr

######################

def entity_linker_types(instance):
  return np.concatenate([
    _entity_linker_types_from_mention(instance.subject_entity),
    _entity_linker_types_from_mention(instance.object_entity)
  ])

def wikidata_properties(instance):
  subject_list = [x["uri"] for x in instance.subject_entity._.uri_candidates]
  object_list = [x["uri"] for x in instance.object_entity._.uri_candidates]
  wikidataProperties = wikidata.get_properties(subject_list,object_list)
  return np.concatenate([
    _global_properties_from_instance(wikidataProperties),
    _specific_properties_from_instance(wikidataProperties)
  ])

def coref_score(instance):
  subject_score = instance.subject_entity._.uri_candidates[0]["coref_score"] if len(instance.subject_entity._.uri_candidates) > 0 else -1.0
  object_score = instance.object_entity._.uri_candidates[0]["coref_score"] if len(instance.object_entity._.uri_candidates) > 0 else -1.0
  return [ subject_score, object_score ]

def el_score(instance):
  subject_score = instance.subject_entity._.uri_candidates[0]["el_score"] if len(instance.subject_entity._.uri_candidates) > 0 else -1.0
  object_score = instance.object_entity._.uri_candidates[0]["el_score"] if len(instance.object_entity._.uri_candidates) > 0 else -1.0
  return [ subject_score, object_score ]

def coref_score_between_subject_object(instance):
  if instance.subject_entity == instance.object_entity:
    return [5.0]
  if instance.subject_entity._.coref_scores and instance.object_entity in instance.subject_entity._.coref_scores:
    return [ instance.subject_entity._.coref_scores[instance.object_entity] ]
  elif instance.object_entity._.coref_scores and instance.subject_entity in instance.object_entity._.coref_scores:
    return [ instance.object_entity._.coref_scores[instance.subject_entity] ]
  return [-5.0]