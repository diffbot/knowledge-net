import os
import json
import itertools
import pathlib
import sys
import yaml
from random import random
from collections import defaultdict
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, stream=sys.stdout)

import evaluator
from instance import Instance
import numpy as np
from run import generate_candidates
from vocab import Vocab
from config import NUMBER_URI_CANDIDATES_TO_CONSIDER, URI_THRESHOLD, MODEL, KNOWLEDGE_NET_DIR, MULTITASK

EVAL_METHODS = ['span_overlap', 'span_exact', 'uri']
TEXT_EVAL_METHODS = ['span_overlap', 'span_exact']
METHOD_TO_USE_FOR_POSITIVES = 'span_exact'
METHOD_TO_USE_FOR_NEGATIVES = 'span_overlap'

pathlib.Path('tmp').mkdir(parents=True, exist_ok=True)
pathlib.Path('vocab').mkdir(parents=True, exist_ok=True)

def is_in_gold(gold_facts, subject_entity, object_entity, property_id):
  ret = dict()
  for method in TEXT_EVAL_METHODS:
    found = []
    my_fact = evaluator.KNFact(None, property_id, 
      subject_entity.start_char, subject_entity.end_char, 
      object_entity.start_char, object_entity.end_char, 
      "", "", str(subject_entity), str(object_entity), None, None)
    for gold_fact in gold_facts:
      if evaluator.twoFactsMatch(gold_fact, my_fact, method):
        found.append(gold_fact)
    ret[method] = found
  return ret

def is_in_gold_uri(gold_facts, subject_entity, object_entity, subject_el_candidate, object_el_candidate, property_id):
  found = []
  my_fact = evaluator.KNFact(None, property_id, 
    subject_entity.start_char, subject_entity.end_char, 
    object_entity.start_char, object_entity.end_char, 
    subject_el_candidate['uri'], object_el_candidate['uri'], str(subject_entity), str(object_entity), None, None)
  if not evaluator.isValidForURI(my_fact):
    return found
  for gold_fact in gold_facts:
    if not evaluator.isValidForURI(gold_fact):
      continue
    if evaluator.twoFactsMatch(gold_fact, my_fact, "uri"):
      found.append(gold_fact)
  return found

def find_sentences_that_overlap(doc, start, end):
  ret = []
  for sent in doc.sents:
    if evaluator.overlap(sent.start_char, sent.end_char, start, end):
      ret.append(sent)
  return ret

def contains_property_id(properties, property_id):
  for p in properties:
    if p.propertyId == property_id:
      return True
  return False

if __name__ == "__main__":
  vocab = Vocab()

  which_model = sys.argv[1] if len(sys.argv) > 1 else "text"
  logging.info("Generating training data for %s", which_model)

  def get_gold_instances(gold_dataset, properties):
    num_total_gold_facts = 0
    num_total_gold_uri_facts = 0
    num_total_facts_found_per_method = dict()
    for method in EVAL_METHODS: 
      num_total_facts_found_per_method[method] = 0
    cont = 0
    for document in gold_dataset:
      cont +=1
      logging.info("Documentid: " + str(document.documentId) + "\t" + str(cont) + " of " + str(len(gold_dataset)))
      candidates = generate_candidates(document.documentText)
      for gold_passage in document.passages:
        annotated_properties = list(map(lambda x: x.propertyId, gold_passage.exhaustivelyAnnotatedProperties))
        if len(annotated_properties) == 0:
          continue
        gold_facts = gold_passage.facts
        gold_count = len(gold_facts)
        actual_count = 0
        gold_facts_found_per_method = dict()
        for method in EVAL_METHODS: 
          gold_facts_found_per_method[method] = set()
        for candidate in candidates:
          if not candidate.is_in_span(gold_passage.passageStart, gold_passage.passageEnd):
            continue
          logging.debug("\t" + str(candidate.subject_entity) + "\t" + str(candidate.object_entity))
          vocab.call(candidate)
          labels = dict()
          for property_id in annotated_properties:
            if not candidate.is_candidate_for(property_id):
              continue
            found_golds_per_method = is_in_gold(gold_facts, candidate.subject_entity, candidate.object_entity, property_id)
            label = None
            if len(found_golds_per_method[METHOD_TO_USE_FOR_POSITIVES]) > 0:
              label = True
            elif len(found_golds_per_method[METHOD_TO_USE_FOR_NEGATIVES]) == 0:
              label = False
            for method in TEXT_EVAL_METHODS: 
              gold_facts_found_per_method[method].update(found_golds_per_method[method])

            uri_label = False
            for uri_candidate in candidate.get_uri_instances()[:NUMBER_URI_CANDIDATES_TO_CONSIDER]:
              #if uri_candidate.subject_entity["score"] < URI_THRESHOLD or uri_candidate.object_entity["score"] < URI_THRESHOLD:
              #  continue
              uri_golds = is_in_gold_uri(gold_facts, candidate.subject_entity, candidate.object_entity, uri_candidate.subject_entity, uri_candidate.object_entity, property_id)
              uri_candidate.labels[property_id] = len(uri_golds) > 0
              if len(uri_golds) > 0:
                uri_label = True
              gold_facts_found_per_method["uri"].update(uri_golds)
            if label is not None:
              if MODEL == "ours" or MODEL == "bert":
                if label:
                  labels[property_id] = 1 if not uri_label else 2
                else:
                  labels[property_id] = 0
              else:
                labels[property_id] = 1 if label else 0
          candidate.labels = labels
          yield candidate

        actual_count = len(gold_facts_found_per_method[METHOD_TO_USE_FOR_POSITIVES])

        if gold_count != actual_count:
          logging.debug("Gold count != Actual count %s %s %s", gold_passage.passageId, gold_count, actual_count)
          logging.debug("Missing %s in %s", list(map(lambda x: x.humanReadable, set(gold_facts) - gold_facts_found_per_method[METHOD_TO_USE_FOR_POSITIVES])), document.documentText)

        num_total_gold_facts += gold_count
        num_total_gold_uri_facts += len(list(filter(lambda x: evaluator.isValidForURI(x), gold_facts)))
        for method in EVAL_METHODS: 
          num_total_facts_found_per_method[method] += len(gold_facts_found_per_method[method])

    #deprecated. use evalute.py with config CANDIDATE_RECALL=True instead
    logging.debug("Candidate recall (span_overlap) = %.3f", float(num_total_facts_found_per_method['span_overlap']) / float(num_total_gold_facts))
    logging.debug("Candidate recall (span_exact) = %.3f", float(num_total_facts_found_per_method['span_exact']) / float(num_total_gold_facts))
    logging.debug("Candidate recall (uri) = %.3f", float(num_total_facts_found_per_method['uri']) / float(num_total_gold_uri_facts))

  def read_kn_file(path, folds):
    ret = []
    properties = dict()
    for fold in folds:
      goldDataset, goldProperties = evaluator.readKnowledgenetFile(path, fold)
      ret.extend(goldDataset.values())
      properties.update(goldProperties)
    return (ret, properties)

  logging.info("Reading KN datasets")
  training_data, properties = read_kn_file(os.path.join(KNOWLEDGE_NET_DIR, 'train.json'), [1,2,3])
  validation_data, _ = read_kn_file(os.path.join(KNOWLEDGE_NET_DIR, 'train.json'), [4])

  train_instances = get_gold_instances(training_data, properties.keys())

  validation_instances = get_gold_instances(validation_data, properties.keys())

  # train models given as args (seperated by commas) or all predicates if none given
  predicates = sys.argv[2].split(",") if len(sys.argv) > 2 else properties.keys()

  import tensorflow as tf

  if which_model == "vocab":
    for _ in train_instances:
      pass
    for _ in validation_instances:
      pass
    vocab.save()
  else:

    def write_sequence_record(numWords, wordFeatures, lm_layers, global_features, writer):
      """Writes a vector as a TFRecord.

      Args:
        vector: A 2D Numpy float array.
        writer: A ``tf.python_io.TFRecordWriter``.
      """

      featureDict = {
        "numWords": tf.train.Feature(int64_list=tf.train.Int64List(value=[numWords])),
        "wordFeatures": tf.train.Feature(float_list=tf.train.FloatList(value=wordFeatures.flatten().tolist())),
      }
      if len(lm_layers) > 0:
        featureDict["lmLayers"] = tf.train.Feature(float_list=tf.train.FloatList(value=list(lm_layers.flatten().tolist())))
      if len(global_features) > 0:
        featureDict["globalFeatures"] = tf.train.Feature(float_list=tf.train.FloatList(value=global_features.flatten().tolist()))
      #for i, f in enumerate(embed):
      #  featureDict['embed'+str(i)] = tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(f,np.int32).flatten().tolist()))

      example = tf.train.Example(features=tf.train.Features(feature=featureDict))

      writer.write(example.SerializeToString())

    def write_header(instance, output_dir, pred_list):
      with open(os.path.join(output_dir, "header.txt"), "w") as writer:
        numWords, wordFeatures, lm_layers, global_features = instance.featurize()
        features = [["wordFeatures", [-1, wordFeatures.shape[1]], "word"]]
        if len(lm_layers) > 0:
          features.append(["lmLayers", [-1, lm_layers.shape[1], lm_layers.shape[2]], "lm"])
        if len(global_features) > 0:
          features.append(["globalFeatures", [global_features.shape[0]], "global"])
        writer.write(yaml.dump(
          {
            "features": features,
            "predicates": list(pred_list),
            "outputs_per_predicate": 2
          }
        ))

    if MULTITASK:
        logging.info("MultiTask")

        output_dir = os.path.join('data', "all")
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        labels_vocab = [
          "0,0",
          "1,0",
          "1,1"
        ]

        def write_features(instances, output):
          logging.info(output)
          with tf.python_io.TFRecordWriter(os.path.join(output_dir, output+".tfrecord")) as writer, open(os.path.join(output_dir, output+".labels"), "w") as labels_writer:
            num = defaultdict(int)
            numPositive = defaultdict(int)
            numPositiveUri = defaultdict(int)
            for instance in instances:
              numWords, wordFeatures, lm_layers, global_features = instance.featurize()
              string_labels = []
              for property_id in predicates:
                if instance.contains_property(property_id):
                  num[property_id] += 1
                  if instance.labels[property_id]:
                    numPositive[property_id] += 1
                  if instance.labels[property_id] == 2:
                    numPositiveUri[property_id] += 1
                  string_labels.append(labels_vocab[instance.labels[property_id]])
                else:
                  string_labels.append("-1,-1")
              write_sequence_record(numWords, wordFeatures, lm_layers, global_features, writer)
              labels_writer.write(",".join(string_labels) + "\n")
            for property_id in predicates:
              logging.info(property_id)
              logging.info("Num %d Positive Text %d Positive Uri %d", num[property_id], numPositive[property_id], numPositiveUri[property_id])

        write_header(next(train_instances), output_dir, predicates)
        write_features(train_instances, 'train')
        write_features(validation_instances, 'validation')

    else:

      for property_id in predicates:
        logging.info("Property %s", property_id)

        output_dir = os.path.join('data', property_id)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

        labels_vocab = [
          "0,0",
          "1,0",
          "1,1"
        ]

        def write_features(instances, output):
          logging.info(output)
          with tf.python_io.TFRecordWriter(os.path.join(output_dir, output+".tfrecord")) as writer, open(os.path.join(output_dir, output+".labels"), "w") as labels_writer:
            num = 0
            numPositive = 0
            numPositiveUri = 0
            for instance in instances:
              if not instance.contains_property(property_id):
                continue
              numWords, wordFeatures, lm_layers, global_features = instance.featurize()
              if instance.labels[property_id]:
                numPositive += 1
              if instance.labels[property_id] == 2:
                numPositiveUri += 1
              write_sequence_record(numWords, wordFeatures, lm_layers, global_features, writer)
              labels_writer.write(labels_vocab[instance.labels[property_id]] + "\n")
              num += 1
          logging.info("Num %d Positive Text %d Positive Uri %d", num, numPositive, numPositiveUri)

        write_header(next(train_instances), output_dir, [property_id])
        write_features(train_instances, 'train')
        write_features(validation_instances, 'validation')
