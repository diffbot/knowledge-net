import tensorflow as tf
import itertools
from os import listdir
import os
import entitylinker
import bert_wrapper
import yaml
import sys

import spacy
nlp = spacy.load('en_core_web_lg')
from spacy.tokens import Span
import numpy as np
from instance import Instance
import collections
from config import MODELS_DIR, MODEL, URI_THRESHOLD, USE_ENTITY_LINKER, USE_BERT, MULTITASK, CANDIDATE_RECALL

PRONOUN_TAGS = set(['PRP','PRP$', 'WDT', 'WP', 'WP$', 'WRB'])
pronoun_lists = {
  'PERSON': ["i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
            "who", "whom", "mine", "yours", "his", "hers", "ours", "theirs", "whose",
            "myself", "yourself", "himself", "herself", "ourselves", "themselves",
            "their", "our", "your", "my"], #person
  'ORG':["it", "its", "we", "they", "us", "them", "ours", "theirs", "whose",
            "itself", "ourselves", "themselves",
            "where", "which", "their", "our"], # org
  'GPE':["where", "there", "it"], #location
  'DATE': ["when", "whenever"], #date
}
all_pronouns = set()
inv_pronoun_map = collections.defaultdict(list)
for k, pronoun_list in pronoun_lists.items():
  for v in pronoun_list:
    inv_pronoun_map[v].append(k)
    all_pronouns.add(v)

path = MODELS_DIR

USE_NOUN_CHUNKS = False

import neuralcoref
neuralcoref.add_to_pipe(nlp)

Span.set_extension("fused_type", default="")
Span.set_extension("is_pronoun", default=False)

models = []
if CANDIDATE_RECALL:
  models.append((None, None, collections.defaultdict(lambda: (0,0))))
elif os.path.exists(path):
  for model_name in listdir(path):
    sess = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(sess, ["serve"], os.path.join(path,model_name))
    #print([n.name for n in sess.graph.as_graph_def().node][:10])
    header_file = os.path.join(path,model_name,"header.txt")
    model_preds = []
    predicate_thresholds = {}
    if os.path.exists(header_file):
      with open(header_file, 'r') as file:
        config = yaml.safe_load(file)
        model_preds = config["predicates"]
        thresholds = config["thresholds"] if "thresholds" in config else {}
        for predicate_id in model_preds:
          threshold = thresholds[predicate_id]["text"] if predicate_id in thresholds and "text" in thresholds[predicate_id] else 1.0
          uri_threshold = thresholds[predicate_id]["uri"] if predicate_id in thresholds and "uri" in thresholds[predicate_id] else 1.0
          predicate_thresholds[predicate_id] = (threshold, uri_threshold)
    
    model = (sess, model_preds, predicate_thresholds)
    models.append(model)

STOP_WORDS = ['the']
TITLES = ['prince','king','queen','princess']

def generate_candidates(text):
  doc = nlp(text)
  instances = []
  all_mentions = []
  mentions_per_sentence = []
  for sentence in doc.sents:
    mentions = []
    # add entities from NER
    for ent in sentence.ents:
      start = ent.start
      end = ent.end
      if ent[0].text.lower() in STOP_WORDS or ent[0].text.lower() in TITLES:
        start += 1
      if ent[-1].text == "." and ent.end == sentence.end:
        end -= 1
      if end > start:
        mention = Span(doc, start, end)
        mention._.fused_type = [ent.label_]
        mentions.append(mention)
    # add pronouns as mentions
    for word in sentence:
      if word.tag_ in PRONOUN_TAGS or word.text.lower() in all_pronouns:
        mention = Span(doc, word.i, word.i+1)
        if mention.text.lower() in inv_pronoun_map:
          mention._.fused_type = inv_pronoun_map[mention.text.lower()]
          mention._.is_pronoun = True
        mentions.append(mention)
    if USE_NOUN_CHUNKS:
      for chunk in sentence.noun_chunks:
        root = chunk.root
        if root.ent_type_ or root.tag_ in PRONOUN_TAGS: # we already have it
          continue
        mentions.append(chunk)
    mentions_per_sentence.append(mentions)
    all_mentions.extend(mentions)
  for i, mentions in enumerate(mentions_per_sentence):
    for (subject_entity, object_entity) in itertools.product(mentions, mentions):
      instances.append(Instance(subject_entity, object_entity))
    if i > 0:
      prev_sentence_mentions = mentions_per_sentence[i-1]
      for (subject_entity, object_entity) in itertools.product(mentions, prev_sentence_mentions):
        instances.append(Instance(subject_entity, object_entity))
      for (subject_entity, object_entity) in itertools.product(prev_sentence_mentions, mentions):
        instances.append(Instance(subject_entity, object_entity))

  if USE_ENTITY_LINKER:
    entitylinker.link(doc, all_mentions)
  if USE_BERT:
    bert_wrapper.run(doc)

  return instances

def classify_instances(instances, predicate_ids_to_classify=None, includeUri=True):
  if len(instances) == 0:
    return
  max_length = max([ len(instance.get_words()) for instance in instances ])
  features_batch = []
  lm_features_batch = []
  len_batch = []
  global_features_batch = []
  if not CANDIDATE_RECALL:
    for instance in instances:
      length, wordFeatures, lm_layers, global_features = instance.featurize()
      if max_length-length < 0:
        print(length, "is greater than", max_length)
      ar = np.pad(np.array(wordFeatures, np.float32), [(0,max_length-length), (0,0)], 'constant')
      features_batch.append(ar)

      if len(lm_layers) > 0:
        lm_ar = np.pad(np.array(lm_layers, np.float32), [(0,max_length-length), (0,0), (0,0)], 'constant')
        lm_features_batch.append(lm_ar)

      len_batch.append(length)

      if len(global_features) > 0:
        global_features_batch.append(global_features)

  for sess, model_preds, predicate_thresholds in models:
    if predicate_ids_to_classify is None:
      predicate_ids_to_classify = predicate_thresholds.keys()
    model_preds = predicate_ids_to_classify if CANDIDATE_RECALL else model_preds
    if len(set(model_preds).intersection(predicate_ids_to_classify)) == 0:
      continue
    features_dict = {
      'numWords:0':np.array(len_batch, np.int32),
      'wordFeatures:0': np.array(features_batch, np.float32), 
    }
    if len(lm_features_batch) > 0:
      features_dict["lmLayers:0"] = np.array(lm_features_batch, np.float32)
    if len(global_features_batch) > 0:
      features_dict["globalFeatures:0"] = np.array(global_features_batch, np.float32)
    if CANDIDATE_RECALL:
      results = [ [1.0]*(len(model_preds)*2) for _ in instances ]
    else:
      results = sess.run('seqclassifier/Sigmoid:0',
                feed_dict=features_dict)
    for instance, label in zip(instances, results):
      for index, predicate_id in enumerate(model_preds):
        if predicate_id not in predicate_ids_to_classify or not instance.is_candidate_for(predicate_id):
          continue
        (threshold, uri_threshold) = predicate_thresholds[predicate_id]
        instance.labels[predicate_id] = label[index*2] > threshold
        instance.scores[predicate_id] = label[index*2]

        if includeUri:
          uri_instances = instance.get_uri_instances()
          if len(uri_instances) > 0:
            if MODEL == "ours" or MODEL == "bert":
              uri_instance = uri_instances[0]
              uri_instance.labels[predicate_id] = label[(index*2)+1] > uri_threshold
              uri_instance.scores[predicate_id] = label[(index*2)+1]
              instance.uri_instances[predicate_id] = uri_instance
            else:
              uri_instance = uri_instances[0]
              uri_instance.labels[predicate_id] = True
              instance.uri_instances[predicate_id] = uri_instance
        

def run_batch(texts):
  ret = []
  for text in texts:
    instances = generate_candidates(text)
    classify_instances(instances)
    ret.append(instances)
  return ret

if __name__ == "__main__":
  if len(models) == 0:
    print("No trained model")
    sys.exit(0)
  import fileinput
  for line in fileinput.input():
    for fact in run_batch([line])[0]:
      if fact.has_positive_label():
        print(fact)