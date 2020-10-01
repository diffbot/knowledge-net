# KnowledgeNet

KnowledgeNet is a benchmark dataset for the task of automatically populating a knowledge base (Wikidata) with facts expressed in natural language text on the web. KnowledgeNet provides text exhaustively annotated with facts, thus enabling the holistic end-to-end evaluation of **knowledge base population systems** as a whole, unlike previous benchmarks that are more suitable for the evaluation of individual subcomponents (e.g., entity linking, relation extraction). 

For instance, the dataset contains text expressing the fact ([Gennaro Basile](https://www.wikidata.org/wiki/Q1367602); RESIDENCE; [Moravia](https://www.wikidata.org/wiki/Q43266)), in the passage:
"Gennaro Basile was an Italian painter, born in Naples but active in the German-speaking countries. He settled at Brünn, in Moravia, and lived about 1756..."

For a description of the dataset and baseline systems, please refer to our [EMNLP paper](https://github.com/diffbot/knowledge-net/blob/master/knowledgenet-emnlp-cameraready.pdf).

To visualize the dataset you can dowload this [file](/human_readable-v10-notest.zip) and open it with a browser. It contains a human readable tabular version of the dataset.


## Leaderboard

|        System       		| Link F1		| Text F1  		|
|---------------------   	| ---------	|--------- 		|
| _Human_	                                                            | **0.822** | **0.878** 	|
| [Diffbot Joint Model](https://www.diffbot.com/)                     | 0.726 		  | 0.810     	|
| [DYGIE++](https://arxiv.org/pdf/1909.03546.pdf)                     | N/A 		  | 0.754     	|
| [Diffbot NER, coref, linker](https://www.diffbot.com/) + [Matching the Blanks (relations)](https://arxiv.org/pdf/1906.03158.pdf)                     | 0.695 		  | 0.737     	|
| [Diffbot Open Relation Extraction (no training data for relations)](https://www.diffbot.com/)                     | 0.546 		  | 0.560     	|
| KnowledgeNet Baseline	5	(Baseline 4 + BERT for relations)                         | 0.504 		| 0.688     	|
| KnowledgeNet Baseline 4	(Baseline 3 + noisy candidate facts)        | 0.491 		| 0.621     	|
| KnowledgeNet Baseline 3	(Baseline 2 + external information)               | 0.362 		| 0.545     	|
| KnowledgeNet Baseline 2	(Baseline 1 + Huggingface coref)       | 0.342 		| 0.554     	|
| KnowledgeNet Baseline 1 (Spacy NER, Diffbot linker, Bi-LSTM for relations)| 0.281 		| 0.518     	|



## Getting Started

The training set is available at `train.json`. Each document contains a number of _passages_. A passage represents a sentence considered for annotation. Here is an example document:

```json
{
  "fold": 2,
  "documentId": "8313",
  "source": "DBpedia Abstract",
  "documentText": "Gennaro Basile\n\nGennaro Basile was an Italian painter, born in Naples but active in the German-speaking countries. He settled at Brünn, in Moravia, and lived about 1756. His best picture is the altar-piece in the chapel of the chateau at Seeberg, in Salzburg. Most of his works remained in Moravia.",
  "passages": [
    {
      "passageId": "8313:16:114",
      "passageStart": 16,
      "passageEnd": 114,
      "passageText": "Gennaro Basile was an Italian painter, born in Naples but active in the German-speaking countries.",
      "exhaustivelyAnnotatedProperties": [
        {          
          "propertyId": "12",
          "propertyName": "PLACE_OF_BIRTH",
          "propertyDescription": "Describes the relationship between a person and the location where she/he was born."
        }
      ],
      "facts": [
        {
          "factId": "8313:16:30:63:69:12",
          "propertyId": "12",
          "humanReadable": "<Gennaro Basile> <PLACE_OF_BIRTH> <Naples>",
          "annotatedPassage": "<Gennaro Basile> was an Italian painter, born in <Naples> but active in the German-speaking countries.",
          "subjectStart": 16,
          "subjectEnd": 30,
          "subjectText": "Gennaro Basile",
          "subjectUri": "http://www.wikidata.org/entity/Q19517888",
          "objectStart": 63,
          "objectEnd": 69,
          "objectText": "Naples",
          "objectUri": "http://www.wikidata.org/entity/Q2634"
        }
      ]
    },
    {
      "passageId": "8313:115:169",
      "passageStart": 115,
      "passageEnd": 169,
      "passageText": "He settled at Brünn, in Moravia, and lived about 1756.",
      "exhaustivelyAnnotatedProperties": [
        {
          "propertyId": "11",
          "propertyName": "PLACE_OF_RESIDENCE",
          "propertyDescription": "Describes the relationship between a person and the location where she/he lives/lived."
        },
        {
          "propertyId": "12",
          "propertyName": "PLACE_OF_BIRTH",
          "propertyDescription": "Describes the relationship between a person and the location where she/he was born."
        }
      ],
      "facts": [
        {
          "factId": "8313:115:117:129:134:11",
          "propertyId": "11",
          "humanReadable": "<He> <PLACE_OF_RESIDENCE> <Brünn>",
          "annotatedPassage": "<He> settled at <Brünn>, in Moravia, and lived about 1756.",
          "subjectStart": 115,
          "subjectEnd": 117,
          "subjectText": "He",
          "subjectUri": "http://www.wikidata.org/entity/Q19517888",
          "objectStart": 129,
          "objectEnd": 134,
          "objectText": "Brünn",
          "objectUri": "http://www.wikidata.org/entity/Q14960"
        },
        {
          "factId": "8313:115:117:139:146:11",
          "propertyId": "11",
          "humanReadable": "<He> <PLACE_OF_RESIDENCE> <Moravia>",
          "annotatedPassage": "<He> settled at Brünn, in <Moravia>, and lived about 1756.",
          "subjectStart": 115,
          "subjectEnd": 117,
          "subjectText": "He",
          "subjectUri": "http://www.wikidata.org/entity/Q19517888",
          "objectStart": 139,
          "objectEnd": 146,
          "objectText": "Moravia",
          "objectUri": "http://www.wikidata.org/entity/Q43266"
        }
      ]
    }
  ]
}
```


### Evaluation
The official evaluation script is also available for download and can be used to evaluate a system using the training set (via cross-validation). The script takes a gold standard file (e.g., `train.json`) and a prediction file (which needs to be produced by the system). The prediction file should look exactly like the gold standard file (same documents and fields), except for the contents of `facts` (which should contain the facts predicted by the system).

```
usage: evaluator.py [-h] [-e {span_exact,span_overlap,uri}] [-c]
                    [-a ANALYSISPATH] [-f {1,2,3,4,5,-1}]
                    goldFile predictionFile

positional arguments:
  goldFile                            path of the KnowledgeNet file with the gold data
  predictionFile                      path of the KnowledgeNet file with the predicted data

optional arguments:
  -h, --help                          show this help message and exit
  -e {span_exact,span_overlap,uri}    choose the evaluation method: span-exact vs span-overlap vs uri
  -c                                  print raw counts of tp/fn/fp for prec/rec/F1 metrics
  -a ANALYSISPATH                     folder to store error analysis and results files
                                      (default=no analysis).
  -f {1,2,3,4,5,-1}                   folds to evaluate. Default is 4. Choose -1 to evaluate on all the folds.
```

The prediction file has to keep the same unique identifiers and attributes for the given documents and passages. 
Each new fact must contain the following attributes that are needed to run the evaluation script: 
* `subjectStart`
* `subjectEnd`
* `objectStart`
* `objectEnd`
* `subjectUri` (can be empty)
* `objectUri`  (can be empty)
* `propertyId`

A `factId` will be automatically generated from these attributes.

#### Evaluation Methods
Two facts are considered the same when they have the same property, and there is a match between the values for subject and object.

We consider three different methods to establish if there is a match:
* **Span Overlap** (`span_overlap`): there is an overlap between the character offsets (default in the evaluation script)
* **Span Exact** (`span_exact`): the character offsets are exactly the same
* **URI** (`uri`): Wikidata URIs are the same (only applies to facts that have URIs for both the subject and the object)

#### Error Analysis
In order to facilitate error analysis the script creates a simple html file for browser visualization. It can be enabled using the option `-a <path to the analysis folder>`.

## Adding a system to the leaderboard

To preserve the integrity of the results, we have released the test set (fifth fold) without annotations (`test-no-facts.json`). To evaluate the results of your system and (optionally) add your system to the leaderboard, please send an email with your prediction file to filipe[at]diffbot[dot]com. 

# Code

[EMNLP Baseline Code](baselines/EMNLP2019/README.md)
