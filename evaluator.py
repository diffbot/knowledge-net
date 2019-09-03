# Official evaluation script for KnowledgeNet.

'''
Command line usage
------------------
usage: evaluator.py [-h] [-e {span_exact,span_overlap,uri}] [-c] [-a ANALYSISPATH] [-f {1,2,3,4,5}]
                    goldFile predictionFile

positional arguments:
  goldFile                Path of the KnowledgeNet file with the gold data
  predictionFile          Path of the KnowledgeNet file with the predicted data

optional arguments:
  -h, --help              show this help message and exit
  -e {span_exact,span_overlap,uri}  Choose the evaluation method: span-exact vs span-overlap vs uri          
  -c                      print raw counts of tp/fn/fp for prec/rec/F1 metrics
  -a ANALYSISPATH         Folder to store error analysis files (default=no analysis).
  -f {1,2,3,4,5}          folds to evaluate (useful during cross-validation). Default is 4.
'''

import copy
import sys
import json
import argparse
import os
from collections import defaultdict
import io

# Internal representation for a Knowledgenet document.
class KNDocument:
  def __init__(self, documentId, documentText, fold, passages, source):
    self.documentId = documentId
    self.documentText = documentText
    self.fold = fold
    self.passages = passages
    self.source = source

  def __hash__(self):
    return hash(self.documentId)

  def __eq__(self, othr):
    return (self.documentId == othr.documentId)

class KNDPassage:
  def __init__(self, passageId, exhaustivelyAnnotatedProperties, passageStart, passageEnd, passageText, facts):
    self.passageId = passageId
    self.exhaustivelyAnnotatedProperties = exhaustivelyAnnotatedProperties
    self.passageStart = passageStart
    self.passageEnd = passageEnd
    self.passageText = passageText
    self.facts = facts

  def __hash__(self):
    return hash(self.passageId)

  def __eq__(self, othr):
    return (self.passageId == othr.passageId)

class KNDProperty:
  def __init__(self, propertyId, propertyName, propertyDescription):
    self.propertyId = propertyId
    self.propertyName = propertyName
    self.propertyDescription = propertyDescription

  def __hash__(self):
    return hash(self.propertyId)

  def __eq__(self, othr):
    return (self.propertyId == othr.propertyId)

class KNFact:
  def __init__(self, factId, propertyId, subjectStart, subjectEnd, objectStart, objectEnd, subjectUri, objectUri, subjectText, 
    objectText, annotatedPassage, humanReadable):
    self.factId = factId
    self.propertyId = propertyId
    self.subjectStart = subjectStart
    self.subjectEnd = subjectEnd
    self.objectStart = objectStart
    self.objectEnd = objectEnd
    self.subjectUri = subjectUri
    self.objectUri = objectUri
    self.subjectText = subjectText
    self.objectText = objectText
    self.annotatedPassage = annotatedPassage
    self.humanReadable = humanReadable

  def __hash__(self):
    return hash(self.factId)

  def __eq__(self, othr):
    return (self.factId == othr.factId)

#######################################################

# Loads Knowledgenet documents
def readKnowledgenetFile(dataset, foldToRead):
  documents = {}
  properties = {}
  reader = io.open(dataset, "r")
  passagesWithError = set()
  for line in reader:
    document = json.loads(line)
    documentId = str(document["documentId"])
    documentText = document["documentText"]
    fold = document["fold"]
    source = "TREx"
    if "source" in document:
      source = document["source"]
    if fold != foldToRead and foldToRead != -1:
      continue
    passages = []
    for passage in document["passages"]:
      passageId = passage["passageId"]
      exhaustivelyAnnotatedProperties = []
      for property in passage["exhaustivelyAnnotatedProperties"]:
        propertyId = property["propertyId"]
        propertyName = property["propertyName"]
        propertyDescription = property["propertyDescription"]
        properties[propertyId] = propertyName
        knpr = KNDProperty(propertyId, propertyName, propertyDescription)
        exhaustivelyAnnotatedProperties.append(knpr)
      passageStart = passage["passageStart"]
      passageEnd = passage["passageEnd"]
      passageText = passage["passageText"]
      facts = []
      for fact in passage["facts"]:
        before = len(passagesWithError)
        # those attributes have to be present and can't be empty
        factId = fact["factId"] if (checkField("factId", fact) == "") else passagesWithError.add(passageId + "\t" + checkField("factId", fact))
        propertyId = fact["propertyId"] if (checkField("propertyId", fact) == "") else passagesWithError.add(passageId + "\t" + checkField("propertyId", fact))
        subjectStart = fact["subjectStart"] if (checkField("subjectStart", fact) == "") else passagesWithError.add(passageId + "\t" + checkField("subjectStart", fact))
        subjectEnd = fact["subjectEnd"] if (checkField("subjectEnd", fact) == "") else passagesWithError.add(passageId + "\t" + checkField("subjectEnd", fact))
        objectStart = fact["objectStart"] if (checkField("objectStart", fact) == "") else passagesWithError.add(passageId + "\t" + checkField("objectStart", fact))
        objectEnd = fact["objectEnd"] if (checkField("objectEnd", fact) == "") else passagesWithError.add(passageId + "\t" + checkField("objectEnd", fact))
        # those attributes have to be present but can be empty
        subjectUri = fact["subjectUri"] if ("subjectUri" in fact) else passagesWithError.add(passageId + "\t" + "subjectUri\tmissing field")
        objectUri = fact["objectUri"] if ("objectUri" in fact) else passagesWithError.add(passageId + "\t"+ "objectUri\tmissing field")
        after = len(passagesWithError)

        # skip the passage if there is an error for one fact
        if (after>before):
          break

        # those attributes are built when reading, if there are not exceptions
        subjectText = fact["subjectText"] if ("subjectText" in fact and fact["subjectText"] != "") else documentText[subjectStart:subjectEnd]
        objectText = fact["objectText"] if ("objectText" in fact and fact["objectText"] != "") else documentText[objectStart:objectEnd]
        annotatedPassage = buildAnnotatedPassage(documentText, passageStart, passageEnd, subjectStart, subjectEnd, objectStart, objectEnd)
        humanReadable = '<%s> <%s> <%s>' % (subjectText, properties[propertyId], objectText)
        
        knf = KNFact(factId, propertyId, subjectStart, subjectEnd, objectStart, objectEnd, subjectUri, objectUri, 
          subjectText, objectText, annotatedPassage, humanReadable)
        facts.append(knf)
      knp = KNDPassage(passageId, exhaustivelyAnnotatedProperties, passageStart, passageEnd, passageText, facts)
      passages.append(knp)
    knd = KNDocument(documentId, documentText, fold, passages, source)
    documents[documentId] = knd
  reader.close()

  if (len(passagesWithError) > 0):
    print("ERROR - Some facts in the following passages miss necessary fields: ")
    for e in passagesWithError:
      print(" --> " + e)
    sys.exit(1)

  return documents, properties

def buildAnnotatedPassage(documentText, passageStart, passageEnd, subjectStart, subjectEnd, objectStart, objectEnd):
  firstEntityStart = subjectStart if (subjectStart<objectStart) else objectStart
  firstEntityEnd = subjectEnd if (subjectEnd<objectEnd) else objectEnd
  secondEntityStart = objectStart if (subjectStart<objectStart) else subjectStart
  secondEntityEnd = objectEnd if (subjectStart<objectStart) else subjectEnd
  s1 = documentText[passageStart:firstEntityStart]
  s2 = documentText[firstEntityStart:firstEntityEnd]
  s3 = documentText[firstEntityEnd:secondEntityStart]
  s4 = documentText[secondEntityStart:secondEntityEnd]
  s5 = documentText[secondEntityEnd:passageEnd]
  annPassage = '%s<%s>%s<%s>%s' % (s1, s2, s3, s4, s5)
  return annPassage

def checkField(nameField, fact):
  error = ""
  if nameField in fact:
    if fact[nameField] == "":
      error = nameField + "\tempty field"
  else:
    error = nameField + "\tmissing field"
  return error


#######################################################

def overlap(start1, end1, start2, end2):
  overlap = False
  #invalid offsets
  if(start1 >= end1 or start2 >= end2):
    overlap = False
  #start is inside
  if((start1 >= start2 and start1 < end2)):
    overlap = True
  if((start2 >= start1 and start2 < end1)):
    overlap = True
  #end is inside
  if((end1 > start2 and end1 <= end2)):
    overlap = True
  if((end2 > start1 and end2 <= end1)):
    overlap = True
  return overlap

def equalsBySpanExact(fact1, fact2):
  subjectExact = (fact1.subjectStart == fact2.subjectStart) and (fact1.subjectEnd == fact2.subjectEnd)
  objectExact = (fact1.objectStart == fact2.objectStart) and (fact1.objectEnd == fact2.objectEnd)
  return (subjectExact and objectExact)

def equalsBySpanOverlap(fact1, fact2):
  subjectOverlap = overlap(fact1.subjectStart, fact1.subjectEnd, fact2.subjectStart, fact2.subjectEnd)
  objectOverlap = overlap(fact1.objectStart, fact1.objectEnd, fact2.objectStart, fact2.objectEnd)
  return (subjectOverlap and objectOverlap)

def equalsByURI(fact1, fact2):
  subjectUriExact = (getWikidataId(fact1.subjectUri) == getWikidataId(fact2.subjectUri))
  objectUriExact = (getWikidataId(fact1.objectUri) == getWikidataId(fact2.objectUri))
  return (subjectUriExact and objectUriExact)

def getWikidataId(uri):
  if uri.endswith("/"):
    uri = uri.rstrip("/")
  uri = uri.split("/")[-1]
  return uri

def twoFactsMatch(fact1, fact2, evaluation):
  entityMatch = False
  if evaluation == 'uri':
    entityMatch = equalsByURI(fact1, fact2)
  if evaluation == 'span_exact':
    entityMatch = equalsBySpanExact(fact1, fact2)
  if evaluation == 'span_overlap':
    entityMatch = equalsBySpanOverlap(fact1, fact2)
  propertyMatch = (fact1.propertyId == fact2.propertyId)
  return entityMatch and propertyMatch

def isValidForURI(fact):
  propertiesNoUri = set(["5", "15", "99"])
  subjectUri = fact.subjectUri
  objectUri = fact.objectUri
  propertyForUri = not fact.propertyId in propertiesNoUri
  valid = False
  if (not "diffbot" in subjectUri) and (subjectUri != "") and (not "diffbot" in objectUri) and (objectUri != ""):
    valid = True
  return valid and propertyForUri

def filterForURIEvaluation(dataset):
  properties = {}
  for documentId in dataset:
    for passage in dataset[documentId].passages:
      localProperties = {}
      for prop in passage.exhaustivelyAnnotatedProperties:
        localProperties[prop.propertyId] = prop.propertyName
      passage.facts = [ x for x in passage.facts if isValidForURI(x)]
      uniqueFacts = set()
      filteredFacts = []
      for f in passage.facts:
        if not (f.subjectUri, f.propertyId, f.objectUri) in uniqueFacts:
          uniqueFacts.add((f.subjectUri, f.propertyId, f.objectUri))
          properties[f.propertyId] = localProperties[f.propertyId]
          filteredFacts.append(f)
      passage.facts = filteredFacts
  return dataset, properties

#######################################################

def printConfusionMatrix(confusionMatrix):
  print("\n ------ Confusion Matrix ------")
  print('%-30s%-15s%-15s%-15s' % ("Property" , "TP", "FP", "FN"))
  print('%-30s%-15s%-15s%-15s' % ("--------" , "--", "--", "--"))
  tp_total = 0
  fp_total = 0
  fn_total = 0
  for p in confusionMatrix:
    values = confusionMatrix[p]
    tp_total+=confusionMatrix[p]["TP"]
    fp_total+=confusionMatrix[p]["FP"]
    fn_total+=confusionMatrix[p]["FN"]
    print('%-30s%-15i%-15i%-15i' % (p , values["TP"], values["FP"], values["FN"]))
  print('%-30s%-15i%-15i%-15i' % ("#global" , tp_total, fp_total, fn_total))

def microEvaluation(confusionMatrix, printCM):
  if printCM:
    printConfusionMatrix(confusionMatrix)
  evals = []
  print("\n ------ Micro Evaluation ------ ")
  print('%-30s%-15s%-15s%-15s' % ("Property" , "Precision", "Recall", "F1"))
  print('%-30s%-15s%-15s%-15s' % ("--------" , "-----", "-----", "-----"))
  for prop in confusionMatrix:
    tp = confusionMatrix[prop]["TP"]
    fp = confusionMatrix[prop]["FP"]
    fn = confusionMatrix[prop]["FN"]
    precision = float(tp)/(tp+fp) if (tp+fp) else 0.0
    recall = float(tp)/(tp+fn) if (tp+fn) else 0.0
    f_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print('%-30s%-15.3f%-15.3f%-15.3f' % (prop, precision, recall, f_score))
    evals.append((prop, precision, recall, f_score))
  return evals

def macroEvaluation(confusionMatrix):
  evals = []
  print("\n ------ Macro Evaluation ------ ")
  print('%-30s%-15s%-15s%-15s' % ("Property" , "Precision", "Recall", "F1"))
  print('%-30s%-15s%-15s%-15s' % ("--------" , "-----", "-----", "-----"))
  tp_total = 0
  fp_total = 0
  fn_total = 0
  for prop in confusionMatrix:
    tp_total+=confusionMatrix[prop]["TP"]
    fp_total+=confusionMatrix[prop]["FP"]
    fn_total+=confusionMatrix[prop]["FN"]
  precision = float(tp_total)/(tp_total+fp_total) if (tp_total+fp_total) else 0.0
  recall = float(tp_total)/(tp_total+fn_total) if (tp_total+fn_total) else 0.0
  f_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
  print('%-30s%-15.3f%-15.3f%-15.3f' % ("#global", precision, recall, f_score))
  evals.append(("#global", precision, recall, f_score))
  # += '%s\t%f\t%f\t%f\n' % ("#global", precision, recall, f_score)
  return evals

def writeAnalysisFile(analysis, analysisPath, evaluation):
  writer = open(analysisPath + "/analysis_"+evaluation+".json", 'w')
  for documentId in analysis:
    writer.write(json.dumps(analysis[documentId], default=lambda o: o.__dict__))
    writer.write("\n")
  writer.flush()
  writer.close()

def writeResultsFile(evals, analysisPath, evaluation):
  writer = open(analysisPath + "/results_"+evaluation+".tsv", 'w')
  for p in evals:
    writer.write('%s\t%f\t%f\t%f\n' % (p[0], p[1], p[2], p[3]))
  writer.flush()
  writer.close()

def writeHtmlFile(analysis, analysisPath, evaluation, goldProperties):
  import codecs
  writer = codecs.open(analysisPath + "/analysis_"+evaluation+".html", "w", "utf-8")
  writer.write("<HTML><HEAD><TITLE>Error analysis</TITLE>" +
    "</HEAD><BODY style=\"margin:20; padding:0;\" BGCOLOR=\"FFFFFF\";>\n<h2><font color=\"red\">Error analysis</font></h2>\n")
  for documentId in analysis:
    writer.write("<div>")
    writer.write("<h3> Document: " + documentId + "</h2>")
    writer.write("<span style=\"white-space: pre-line\">" + analysis[documentId].documentText + "</span>")

    for passage in analysis[documentId].passages:
      writer.write("<div style=\"padding-top:0px;margin-top:10px;margin-left:3em;padding-left:10px;padding-right:10px;padding-bottom:10px; margin-bottom:10px; border: 1px dashed #000\">")
      writer.write("<h4> Passage: " + passage.passageId + "</h4>")
      writer.write("<h5> Annotated For: " + ', '.join([(p.propertyName + " (" + p.propertyId + ")") for p in passage.exhaustivelyAnnotatedProperties]) + "</h5>")
      writer.write("<table style=\"width:100%; table-layout:fixed;\">")
      writer.write("<tr>")

      writer.write("<td style=\"vertical-align:top;\">")
      writer.write("<table style=\"width:100%;\">")
      writer.write("<tr>")
      writer.write("<th colspan=\"3\">True Positive</th>")
      writer.write("</tr>")
      for fact in passage.facts:
        if fact.eval == "TP":
          color = "green";
          p = fact.propertyId
          if evaluation == 'uri':
            subject = "<a href=" + fact.subjectUri + ">" + getWikidataId(fact.subjectUri) + "</a>"
            object = "<a href=" + fact.objectUri + ">" + getWikidataId(fact.objectUri) + "</a>"
          else:
            subject = fact.subjectText + " (" + str(fact.subjectStart) + "-" + str(fact.subjectEnd) + ")"
            object = fact.objectText + " (" + str(fact.objectStart) + "-" + str(fact.objectEnd) + ")"
          writer.write("<tr><td style=\"border:1px dashed "+color+"; text-align: center; vertical-align: middle;\">" + subject + "<br>" + 
            #"<a href=" + fact.subjectUri + ">" + getWikidataId(fact.subjectUri) + "</a>" + 
            "</td><td style=\"border:1px dashed "+color+"; text-align: center; vertical-align: middle;\">" + p +
            "</td><td style=\"border:1px dashed "+color+"; text-align: center; vertical-align: middle;\">" + object + "<br>" +
            #"<a href=" + fact.objectUri + ">" + getWikidataId(fact.objectUri) + "</a>" +
            "</td></tr>")
      writer.write("</table>")
      writer.write("</td>")
      
      writer.write("<td style=\"vertical-align:top;\">")
      writer.write("<table style=\"width:100%; table-layout:fixed;\">")
      writer.write("<tr>");
      writer.write("<th colspan=\"3\">False Negative</th>");
      writer.write("</tr>");
      for fact in passage.facts:
        if fact.eval == "FN":
          color = "red";
          p = fact.propertyId
          if evaluation == 'uri':
            subject = "<a href=" + fact.subjectUri + ">" + getWikidataId(fact.subjectUri) + "</a>"
            object = "<a href=" + fact.objectUri + ">" + getWikidataId(fact.objectUri) + "</a>"
          else:
            subject = fact.subjectText + " (" + str(fact.subjectStart) + "-" + str(fact.subjectEnd) + ")"
            object = fact.objectText + " (" + str(fact.objectStart) + "-" + str(fact.objectEnd) + ")"
          writer.write("<tr><td style=\"border:1px dashed "+color+"; text-align: center; vertical-align: middle;\">" + subject + "<br>" + 
            #"<a href=" + fact.subjectUri + ">" + getWikidataId(fact.subjectUri) + "</a>" + 
            "</td><td style=\"border:1px dashed "+color+"; text-align: center; vertical-align: middle;\">" + p +
            "</td><td style=\"border:1px dashed "+color+"; text-align: center; vertical-align: middle;\">" + object + "<br>" +
            #"<a href=" + fact.objectUri + ">" + getWikidataId(fact.objectUri) + "</a>" +  
            "</td></tr>")
      writer.write("</table>")
      writer.write("</td>")

      writer.write("<td style=\"vertical-align:top;\">")
      writer.write("<table style=\"width:100%; table-layout:fixed;\">")
      writer.write("<tr>");
      writer.write("<th colspan=\"3\">False Positive</th>");
      writer.write("</tr>");
      for fact in passage.facts:
        if fact.eval == "FP":
          color = "DarkMagenta";
          p = fact.propertyId
          if evaluation == 'uri':
            subject = "<a href=" + fact.subjectUri + ">" + getWikidataId(fact.subjectUri) + "</a>"
            object = "<a href=" + fact.objectUri + ">" + getWikidataId(fact.objectUri) + "</a>"
          else:
            subject = fact.subjectText + " (" + str(fact.subjectStart) + "-" + str(fact.subjectEnd) + ")"
            object = fact.objectText + " (" + str(fact.objectStart) + "-" + str(fact.objectEnd) + ")"
          writer.write("<tr><td style=\"border:1px dashed "+color+"; text-align: center; vertical-align: middle;\">" + subject + "<br>" + 
            #"<a href=" + fact.subjectUri + ">" + getWikidataId(fact.subjectUri) + "</a>" + 
            "</td><td style=\"border:1px dashed "+color+"; text-align: center; vertical-align: middle;\">" + p +
            "</td><td style=\"border:1px dashed "+color+"; text-align: center; vertical-align: middle;\">" + object + "<br>" +
            #"<a href=" + fact.objectUri + ">" + getWikidataId(fact.objectUri) + "</a>" +  
            "</td></tr>")
      writer.write("</table>")
      writer.write("</td>")

      writer.write("</tr>")
      writer.write("</table>")

      writer.write("<h5> Text Passage: </h5>")
      writer.write("<span style=\"white-space: pre-line\">" + passage.passageText + "</span>")
      writer.write("</div>")
    writer.write("</div>")
    writer.write("<hr style=\"border: none;\">")
  writer.write("</BODY></HTML>")
  writer.flush()
  writer.close()

#######################################################

def isFactInSet(fact1, set, evaluation):
  '''
  We re-iterate the list instead of using a set to ensure that the check is done with the same matching method (i.e. span, uri).
  '''
  exists = False
  for fact2 in set:
    exists = twoFactsMatch(fact1, fact2, evaluation)
    if exists:
      break
  return exists

def evaluate(goldDataset, predictionDataset, evaluation, goldProperties):
  confusionMatrix = defaultdict(lambda : defaultdict(lambda : 0))
  predictionDataset_notMatched = copy.deepcopy(predictionDataset)
  analysisDataset = copy.deepcopy(goldDataset)

  for documentId in goldDataset:
    for goldPassage in goldDataset[documentId].passages:
      for goldFact in goldPassage.facts:
        matchedFacts = []
        
        if documentId in predictionDataset:
          if goldPassage in predictionDataset[documentId].passages:
            # find the right passage
            predictionPassage = next(p for p in predictionDataset[documentId].passages if p.passageId == goldPassage.passageId)
            for predictedFact in predictionPassage.facts:
              if twoFactsMatch(goldFact, predictedFact, evaluation): 
                matchedFacts.append(predictedFact)
          else:
            print("ERROR - Can't find passage " + goldPassage.passageId + " in the predicted file.")
            sys.exit(1)
        else:
          print("ERROR - Can't find document " + documentId + " in the predicted file.")
          sys.exit(1)
        
        if len(matchedFacts) == 0:
          confusionMatrix[goldProperties[goldFact.propertyId]]["FN"] += 1

          # FN --> add label FN in the relative fact for analysis
          analysisPassage = next(p for p in analysisDataset[documentId].passages if p.passageId == goldPassage.passageId)
          analysisFact = next(f for f in analysisPassage.facts if f.factId == goldFact.factId)
          analysisFact.eval = "FN"

        else:
          confusionMatrix[goldProperties[goldFact.propertyId]]["TP"] += 1

          # update not matched facts
          # do not replicate the existence check, already did above
          notMatchedPassage = next(p for p in predictionDataset_notMatched[documentId].passages if p.passageId == goldPassage.passageId)
          notMatchedPassage.facts = [ f for f in notMatchedPassage.facts if not isFactInSet(f, matchedFacts, evaluation)]

          # TP --> add label TP in the relative fact for analysis
          analysisPassage = next(p for p in analysisDataset[documentId].passages if p.passageId == goldPassage.passageId)
          analysisFact = next(f for f in analysisPassage.facts if f.factId == goldFact.factId)
          analysisFact.eval = "TP"

  for documentId in predictionDataset_notMatched:
    for notMatchedPassage in predictionDataset_notMatched[documentId].passages:
      for notMatchedFact in notMatchedPassage.facts:
        if documentId in goldDataset:
          if notMatchedPassage in goldDataset[documentId].passages:
            goldPassage = next(p for p in goldDataset[documentId].passages if p.passageId == notMatchedPassage.passageId)
            # skip if not known property
            if not notMatchedFact.propertyId in goldProperties:
              continue
            if notMatchedFact.propertyId in set([i.propertyId for i in goldPassage.exhaustivelyAnnotatedProperties]):
              confusionMatrix[goldProperties[notMatchedFact.propertyId]]["FP"] += 1
          
              # FP --> add facts and label them FP in the analysis
              analysisPassage = next(p for p in analysisDataset[documentId].passages if p.passageId == notMatchedPassage.passageId)
              analysisFact = copy.deepcopy(notMatchedFact)
              analysisFact.eval = "FP"
              analysisPassage.facts.append(analysisFact)
          else:
            print("ERROR - Passage " + notMatchedPassage.passageId + " is not the gold file.")
            sys.exit(1)
        else:
          print("ERROR - Document " + documentId + " is not the gold file.")
          sys.exit(1)

  return confusionMatrix, analysisDataset


#######################################################

def main():
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("goldFile", type=str,
            help="Path of the KnowledgeNet file with the gold data")
  parser.add_argument("predictionFile", type=str,
            help="Path of the KnowledgeNet file with the predicted data")
  parser.add_argument("-e", choices=['span_exact', 'span_overlap', 'uri'], default=None,
            help="Choose the evaluation method: span-exact vs span-overlap vs uri")
  parser.add_argument("-c", default=False, action="store_true",
            help="print raw counts of tp/fn/fp for prec/rec/F1 metrics")
  parser.add_argument("-a", action='store', default="", dest='analysisPath',
                      help='Folder to store error analysis and results files (default=no analysis).')
  parser.add_argument('-f', choices=[1,2,3,4,5], default=4, type=int,
            help='folds to evaluate. Default is 4')

  args = parser.parse_args()

  if (args.e == None):
    print("ERROR - No Evaluation method selected.")
    print("Add one of the following options:")
    print("\t[-e span_exact] (for span-exact evaluation)")
    print("\t[-e span_overlap] (for span-overlap evaluation)")
    print("\t[-e uri] (for uri-evaluation)")
    sys.exit(1)

  # Read files
  gold, goldProperties = readKnowledgenetFile(args.goldFile, args.f)
  prediction, predictedProperties = readKnowledgenetFile(args.predictionFile, args.f)

  # Check validity of the two datasets
  if (len(gold) == 0):
    print("ERROR - No documents in gold_file for fold = " + str(args.f))
  if (len(prediction) == 0):
    print("ERROR - No documents in prediction_file for fold = " + str(args.f))
  if (len(gold) == 0 or len(prediction) == 0):
    sys.exit(1)

  # Filter both datasets if uri only (and update list of properties)
  if args.e == 'uri':
    gold, goldProperties = filterForURIEvaluation(gold)
    prediction, predictedProperties = filterForURIEvaluation(prediction)

  # Evaluate
  confusionMatrix, analysis = evaluate(gold, prediction, args.e, goldProperties)

  # Print results
  print("RESULTS:")
  
  evals = microEvaluation(confusionMatrix, args.c)
  evals.extend(macroEvaluation(confusionMatrix))

  if args.analysisPath != "":
    if not os.path.exists(args.analysisPath):
      os.makedirs(args.analysisPath)
    writeAnalysisFile(analysis, args.analysisPath, args.e)
    writeHtmlFile(analysis, args.analysisPath, args.e, goldProperties)
    writeResultsFile(evals, args.analysisPath, args.e)

if __name__ == "__main__":
    main()

