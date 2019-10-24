import os
import copy
import collections
import sys
import yaml
from os import listdir
from config import MODEL, KNOWLEDGE_NET_DIR
from prepare_data import is_in_gold, is_in_gold_uri

import run
import evaluator

filename = "train.json"

from config import MODELS_DIR
path = MODELS_DIR

# Find threshold for models given as args (seperated by commas) or all models if none given
models = set(sys.argv[1].split(",")) if len(sys.argv) > 1 else set(listdir(path))

all_model_predicates = set()
model_preds = {}
for model in models:
  header_file = os.path.join(path,model,"header.txt")
  with open(header_file, 'r') as file:
    config = yaml.safe_load(file)
    model_preds[model] = config["predicates"]
    all_model_predicates.update(config["predicates"])

gold_dataset, properties = evaluator.readKnowledgenetFile(os.path.join(KNOWLEDGE_NET_DIR, filename), 4)
dataset = copy.deepcopy(gold_dataset)

per_predicate_instances = collections.defaultdict(list)
per_predicate_uri_instances = collections.defaultdict(list)

cont = 0
length = len(dataset.values())
for document in dataset.values():
  cont +=1
  print ("Documentid: " + str(document.documentId) + "\t" + str(cont) + " of " + str(length))
  instances = run.generate_candidates(document.documentText)
  for passage in document.passages:
    annotated_properties = set(map(lambda x: x.propertyId, passage.exhaustivelyAnnotatedProperties))
    if len(annotated_properties) == 0:
      continue
    passage_instances = list(filter(lambda x: x.is_in_span(passage.passageStart, passage.passageEnd), instances))
    if len(passage_instances) == 0 or len(annotated_properties.intersection(all_model_predicates)) == 0:
      continue
    run.classify_instances(passage_instances, annotated_properties.intersection(all_model_predicates))
      
    gold_facts = list(filter(lambda x: x.propertyId in all_model_predicates, passage.facts))
    gold_facts_found = set()
    for fact in passage_instances:
      for predicate_id, score in fact.scores.items():
        gold_facts_matching = is_in_gold(gold_facts, fact.subject_entity, fact.object_entity, predicate_id)["span_overlap"]
        gold_facts_found.update(gold_facts_matching)
        per_predicate_instances[predicate_id].append((score, len(gold_facts_matching) > 0))
    gold_facts_with_zero = set(gold_facts) - gold_facts_found
    for fact in gold_facts_with_zero:
      per_predicate_instances[fact.propertyId].append((0.0, True))

    gold_facts_uri = list(filter(lambda x: evaluator.isValidForURI(x), gold_facts))
    gold_facts_found_uri = set()
    for fact in passage_instances:
      for predicate_id, uri_instance in fact.uri_instances.items():
        if predicate_id not in uri_instance.scores:
          continue
        score = uri_instance.scores[predicate_id]
        gold_facts_matching = is_in_gold_uri(gold_facts_uri, fact.subject_entity, fact.object_entity, uri_instance.subject_entity, uri_instance.object_entity, predicate_id)
        gold_facts_found_uri.update(gold_facts_matching)
        per_predicate_uri_instances[predicate_id].append((score, fact.scores[predicate_id], len(gold_facts_matching) > 0))
    gold_facts_with_zero_uri = set(gold_facts_uri) - gold_facts_found_uri
    for fact in gold_facts_with_zero_uri:
      per_predicate_uri_instances[fact.propertyId].append((0.0, 0.0, True))

def save_threshold(predicate_id, results, which_threshold):
  # Sort results by score and do a single pass to find best threshold
  sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
  precisionAtBest = 0.0
  recallAtBest = 0.0
  best_fscore = 0.0
  best_threshold = 0.95

  tp = 0
  fp = 0
  fn = len(list(filter( lambda x: x[1], sorted_results)))
  if fn == 0:
    return float(best_threshold)
  for i in range(len(sorted_results)):
    r = sorted_results[i]
    
    threshold = r[0]
    label = r[1]
    if label:
      tp += 1
      fn -= 1
    else:
      fp += 1

    if i+1 < len(sorted_results) and sorted_results[i+1][0] == threshold:
      continue

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    if tp == 0:
      continue

    fscore = 2 * precision * recall / (precision + recall)
    if fscore > best_fscore:
      best_fscore = fscore
      best_threshold = (threshold + sorted_results[i+1][0]) / 2.0 if i+1 < len(sorted_results) else threshold
      precisionAtBest = precision
      recallAtBest = recall
  print(predicate_id)
  print("Threshold", best_threshold)
  print("F1", best_fscore)
  print("Precision", precisionAtBest)
  print("Recall", recallAtBest)
  for model, preds in model_preds.items():
    if predicate_id not in preds:
      continue
    with open(os.path.join(path,model,"header.txt"), "r") as text_file:
      config = yaml.safe_load(text_file)
    if "thresholds" not in config:
      config["thresholds"] = {}
    if predicate_id not in config["thresholds"]:
      config["thresholds"][predicate_id] = {}
    config["thresholds"][predicate_id][which_threshold] = float(best_threshold)
    with open(os.path.join(path,model,"header.txt"), "w") as text_file:
      text_file.write(yaml.dump(config))
  return float(best_threshold)

for predicate_id in per_predicate_instances.keys():
  text_threshold = save_threshold(predicate_id, per_predicate_instances[predicate_id], "text")
  if MODEL == "ours" or MODEL == "bert":
    def filter_uri_scores(t):
      (uri_score, text_score, label) = t
      return (0.0, label) if text_score < text_threshold else (uri_score, label)
    uri_results = map(filter_uri_scores, per_predicate_uri_instances[predicate_id])
    save_threshold(predicate_id, uri_results, "uri")