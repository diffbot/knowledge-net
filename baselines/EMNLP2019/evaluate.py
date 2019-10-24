import os
import copy
import sys

import run
import evaluator
from config import KNOWLEDGE_NET_DIR

which = sys.argv[1] if len(sys.argv) > 1 else "dev"

if which == "dev":
  filename = "train.json"
  fold = 4
elif which == "test":
  filename = "test-no-facts.json"
  fold = 5
else:
  sys.exit('Invalid evaluation set')


gold_dataset, properties = evaluator.readKnowledgenetFile(os.path.join(KNOWLEDGE_NET_DIR, filename), fold)
dataset = copy.deepcopy(gold_dataset)
cont = 0

for document in dataset.values():
  cont +=1
  print ("Documentid: " + str(document.documentId) + "\t" + str(cont) + " of " + str(len(dataset.values())))
  instances = run.generate_candidates(document.documentText)
  for passage in document.passages:
    annotated_properties = set(map(lambda x: x.propertyId, passage.exhaustivelyAnnotatedProperties))
    if which == "dev" and len(annotated_properties) == 0:
      continue
    passage_instances = list(filter(lambda x: x.is_in_span(passage.passageStart, passage.passageEnd), instances))
    if which == "dev":
      run.classify_instances(passage_instances, annotated_properties)
    else:
      run.classify_instances(passage_instances)
    passage.facts = []
    for fact in passage_instances:
      for predicate_id, label in fact.labels.items():
        predicate = evaluator.KNDProperty(predicate_id, None, None)
        if not label:
          continue
        passage.facts.append(evaluator.KNFact(None, predicate_id, 
          fact.subject_entity.start_char, fact.subject_entity.end_char, 
          fact.object_entity.start_char, fact.object_entity.end_char, 
          fact.get_subject_uri(predicate_id), fact.get_object_uri(predicate_id), str(fact.subject_entity), str(fact.object_entity), None, None))
# Evaluate
def print_evaluation(eval_type):
  gold = copy.deepcopy(gold_dataset)
  prediction = copy.deepcopy(dataset)
  if eval_type == "uri":
    gold, goldProperties = evaluator.filterForURIEvaluation(gold)
    prediction, _ = evaluator.filterForURIEvaluation(prediction)
  else:
    goldProperties = properties
  confusionMatrix, analysis = evaluator.evaluate(gold, prediction, eval_type, goldProperties)

  # Print results
  print("RESULTS FOR",eval_type)

  evals = evaluator.microEvaluation(confusionMatrix, True)
  evals.extend(evaluator.macroEvaluation(confusionMatrix))
  
  evaluator.writeAnalysisFile(analysis, 'tmp', eval_type)
  evaluator.writeHtmlFile(analysis, 'tmp', eval_type, goldProperties)

print_evaluation("span_overlap")
print_evaluation("uri")
print_evaluation("span_exact")
