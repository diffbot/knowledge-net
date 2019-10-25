import itertools
import numpy as np
import evaluator
import networkx as nx
import features
import global_features
import config
from config import MODEL
import os
from uri_instance import UriInstance


FILTER_ON_TYPES = config.MODEL != "ours" and config.MODEL != "bert"

IOB_TAGS = ['B','I','O']
TYPE_TAGS = ['', 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

EXPECTED_TYPES_PER_PROPERTY = {
  '1':  ('ORG','ORG'), #SUBSIDIARY_OF
  '2':  ('ORG','PERSON'), #FOUNDED_BY
  '3':  ('PERSON','ORG'), #EMPLOYEE_OR_MEMBER_OF
  '4':  ('ORG','PERSON'), #CEO
  '5':  ('ORG','DATE'), #DATE_FOUNDED
  '6':  ('ORG','GPE'), #HEADQUARTERS
  '9':  ('PERSON','ORG'), #EDUCATED_AT
  '10': ('PERSON','NORP'), #NATIONALITY
  '11': ('PERSON','GPE'), #PLACE_OF_RESIDENCE
  '12': ('PERSON','GPE'), #PLACE_OF_BIRTH
  '14': ('PERSON','DATE'), #DATE_OF_DEATH
  '15': ('PERSON','DATE'), #DATE_OF_BIRTH
  '25': ('PERSON','PERSON'), #SPOUSE
  '34': ('PERSON','PERSON'), #CHILD_OF
  '45': ('PERSON','ORG'), #POLITICAL_AFFILIATION
}

PROPERTY_NAMES = {
  '1':  'SUBSIDIARY_OF',
  '2':  'FOUNDED_BY',
  '3':  'EMPLOYEE_OR_MEMBER_OF',
  '4':  'CEO',
  '5':  'DATE_FOUNDED',
  '6':  'HEADQUARTERS',
  '9':  'EDUCATED_AT',
  '10': 'NATIONALITY',
  '11': 'PLACE_OF_RESIDENCE',
  '12': 'PLACE_OF_BIRTH',
  '14': 'DATE_OF_DEATH',
  '15': 'DATE_OF_BIRTH',
  '25': 'SPOUSE',
  '34': 'CHILD_OF',
  '45': 'POLITICAL_AFFILIATION'
}

class Instance:
  def __init__(self, subject_entity, object_entity, labels=None):
    self.subject_entity = subject_entity
    self.object_entity = object_entity
    self.labels = dict() if labels is None else labels
    self.scores = dict()
    self.uri_instances = dict()

  def has_positive_label(self):
    return True in self.labels.values()

  def positive_labels(self):
    return [ k for k,v in self.labels.items() if v ]

  def contains_property(self, property_id):
    return property_id in self.labels

  def is_in_span(self, start_offset, end_offset):
    return (evaluator.overlap(start_offset, end_offset, self.subject_entity.start_char, self.subject_entity.end_char) and
            evaluator.overlap(start_offset, end_offset, self.object_entity.start_char, self.object_entity.end_char))

  def is_candidate_for(self, property_id):
    if not FILTER_ON_TYPES:
      return True
    expected_types = EXPECTED_TYPES_PER_PROPERTY[property_id]
    return expected_types[0] in self.subject_entity._.fused_type and expected_types[1] in self.object_entity._.fused_type

  def featurize(self):
    lm_feature = None
    if MODEL == "simple_pipeline":
      features_to_use = [
        features.embedding(self),
        features.ner_types(self),
        features.subject_position(self),
        features.object_position(self),
        features.dep_path(self, coref=False),
      ]
    elif MODEL == "bert":
      features_to_use = [
        #features.bert_embedding(self),
        features.embedding(self),
        features.ner_types(self),
        features.coref_scores(self),
        features.subject_position(self),
        features.object_position(self),
        features.dep_path(self),
        features.cluster_positions(self),
      ]
      lm_feature = features.bert_layers(self)
    else:
      features_to_use = [
        features.embedding(self),
        features.ner_types(self),
        features.coref_scores(self),
        features.subject_position(self),
        features.object_position(self),
        features.dep_path(self),
        features.cluster_positions(self),
      ]
    features_to_embed = [
    ]
    if MODEL == "ours" or MODEL == "bert":
      global_features_to_use = np.concatenate([
        global_features.entity_linker_types(self),
        global_features.wikidata_properties(self),
        global_features.coref_score(self),
        global_features.el_score(self),
        global_features.coref_score_between_subject_object(self),
        features.ner_types(self)(self.subject_entity.root),
        features.ner_types(self)(self.object_entity.root),
      ])
    elif MODEL == "best_pipeline":
      global_features_to_use = np.concatenate([
        global_features.entity_linker_types(self),
        global_features.wikidata_properties(self),
      ])
    else:
      global_features_to_use = np.array([])
    return self.featurize_sentence_words(features_to_use, features_to_embed, lm_feature) + (global_features_to_use,)

  def get_words(self):
    subject_entity = self.subject_entity
    object_entity = self.object_entity

    first_entity = subject_entity if subject_entity.start < object_entity.start else object_entity
    last_entity = object_entity if subject_entity.start < object_entity.start else subject_entity
    if subject_entity.sent == object_entity.sent:
      words = subject_entity.sent
    else:
      doc = subject_entity.doc
      words = doc[first_entity.sent.start:last_entity.sent.end]
    return words
  def featurize_sentence_words(self, features, features_to_embed, lm_feature):
    words = self.get_words()

    features_vec = [ np.concatenate([ feature(word) for feature in features ]) for word in words ]
    feature_indices_vec = [] #TODO fix this used to embed
    for wordi, word in enumerate(words):
      v = [ f for feature in features_to_embed for f in feature(word) ]
      if len(feature_indices_vec) == 0:
        for i in range(len(v)):
          feature_indices_vec.append([0] * len(words))
      for i in range(len(v)):
        feature_indices_vec[i][wordi] = v[i]
    #if np.array(features_vec).shape[0] != np.array(feature_indices_vec[0]).shape[0]:
    #  print("Num words not equal", np.array(features_vec).shape[0], np.array(feature_indices_vec[0]).shape[0])
    lm_vec = [ lm_feature(word) for word in words ] if lm_feature else []
    
    return (len(words), np.array(features_vec, np.float32), np.array(lm_vec, np.float32))

  def get_uri_instances(self):
    ret = []
    for subject_uri_cand in self.subject_entity._.uri_candidates:
      for object_uri_cand in self.object_entity._.uri_candidates:
        ret.append(UriInstance(subject_uri_cand, object_uri_cand, self))
    return ret

  def get_subject_uri(self, predicate_id):
    return self.uri_instances[predicate_id].subject_entity["uri"] if predicate_id in self.uri_instances and self.uri_instances[predicate_id].labels[predicate_id] else ""

  def get_object_uri(self, predicate_id):
    return self.uri_instances[predicate_id].object_entity["uri"] if predicate_id in self.uri_instances and self.uri_instances[predicate_id].labels[predicate_id] else ""

  def __repr__(self):
    return str(self)

  def __str__(self):
    facts = []
    for predicate_id in self.positive_labels():
        score = self.scores[predicate_id]
        facts.append("{0}|{1} {2}({3:.2f}) {4}|{5}".format(self.subject_entity.text, self.get_subject_uri(predicate_id), PROPERTY_NAMES[predicate_id], score, self.object_entity.text, self.get_object_uri(predicate_id)))
    return "\n".join(facts)