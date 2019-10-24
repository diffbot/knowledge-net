import itertools
import numpy as np
import evaluator
import config
import os
import uri_features

class UriInstance:
  def __init__(self, subject_entity, object_entity, text_instance, labels=None):
    self.subject_entity = subject_entity
    self.object_entity = object_entity
    self.text_instance = text_instance
    self.labels = dict() if labels is None else labels
    self.scores = dict()

  def has_positive_label(self):
    return True in self.labels.values()

  def positive_labels(self):
    return [ k for k,v in self.labels.items() if v ]

  def contains_property(self, property_id):
    return property_id in self.labels

  def featurize(self, property_id):
    features_to_use = [
      uri_features.entity_linker_types,
      uri_features.coref_score,
      uri_features.el_score,
      uri_features.text_score,
    ]
    return np.concatenate([ f(self, property_id) for f in features_to_use ])

  def __repr__(self):
    return str(self)

  def __str__(self):
    def entity_str(e):
      return e["uri"]
    preds = [ "{0}({1:.2f})".format(pred, self.scores[pred]) for pred in self.positive_labels() ]
    return entity_str(self.subject_entity) + " \ " + ",".join(preds)  + " \ " + entity_str(self.object_entity)