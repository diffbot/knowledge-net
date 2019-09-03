# KnowledgeNet

KnowledgeNet is a benchmark dataset for the task of automatically populating a knowledge base (Wikidata) with facts expressed in natural language text on the web. KnowledgeNet provides text exhaustively annotated with facts, thus enabling the holistic end-to-end evaluation of **knowledge base population systems (KBP)** as a whole, unlike previous benchmarks that are more suitable for the evaluation of individual subcomponents (e.g., entity linking, relation extraction). 

For instance, the dataset contains text expressing the fact ([Gennaro Basile](https://www.wikidata.org/wiki/Q1367602); RESIDENCE; [Moravia](https://www.wikidata.org/wiki/Q43266)), in the passage:
"Gennaro Basile was an Italian painter, born in Naples but active in the German-speaking countries. He settled at Brünn, in Moravia, and lived about 1756..."

For a description of the dataset and baseline systems, please refer to our EMNLP paper.


## Leaderboard

|        System       		| Link F1	| Text F1  |
|---------------------   	| ---------	|--------- |
| _Human_					| **0.82** 	| **0.88** |
| KnowledgeNet Baseline	5	| 0.50 		| 0.69     |
| KnowledgeNet Baseline 4	| 0.49 		| 0.62     |
| KnowledgeNet Baseline 3	| 0.36 		| 0.55     |
| KnowledgeNet Baseline 2	| 0.34 		| 0.55     |
| KnowledgeNet Baseline 1	| 0.28 		| 0.52     |



## Getting Started

The training set is available at `train.json`. Here is an example document:

```json
{
  "fold": 2,
  "documentId": "8313",
  "documentText": "Gennaro Basile was an Italian painter, born in Naples but active in the German-speaking countries. He settled at Brünn, in Moravia, and lived about 1756. His best picture is the altar-piece in the chapel of the chateau at Seeberg, in Salzburg. Most of his works remained in Moravia.",
  "passages": [
    {
      "passageId": "8313:0:98",
      "exhaustivelyAnnotatedProperties": [
        {
          "propertyId": "12",
          "propertyName": "PLACE_OF_BIRTH",
          "propertyDescription": "Describes the relationship between a person and the location where she/he was born."
        }
      ],
      "passageStart": 0,
      "passageEnd": 98,
      "passageText": "Gennaro Basile was an Italian painter, born in Naples but active in the German-speaking countries.",
      "facts": [
        {
          "factId": "8313:0:14:47:53:12",
          "propertyId": "12",
          "humanReadable": "<Gennaro Basile> <PLACE_OF_BIRTH> <Naples>",
          "annotatedPassage": "<Gennaro Basile> was an Italian painter, born in <Naples> but active in the German-speaking countries.",
          "subjectStart": 0,
          "subjectEnd": 14,
          "subjectText": "Gennaro Basile",
          "subjectUri": "http://www.wikidata.org/entity/Q19517888",
          "objectStart": 47,
          "objectEnd": 53,
          "objectText": "Naples",
          "objectUri": "http://www.wikidata.org/entity/Q2634"
        }
      ]
    },
    {
      "passageId": "8313:99:153",
      "exhaustivelyAnnotatedProperties": [
        {
          "propertyId": "12",
          "propertyName": "PLACE_OF_BIRTH",
          "propertyDescription": "Describes the relationship between a person and the location where she/he was born."
        },
        {
          "propertyId": "11",
          "propertyName": "PLACE_OF_RESIDENCE",
          "propertyDescription": "Describes the relationship between a person and the location where she/he lives/lived."
        }
      ],
      "passageStart": 99,
      "passageEnd": 153,
      "passageText": "He settled at Brünn, in Moravia, and lived about 1756.",
      "facts": [
        {
          "factId": "8313:99:101:113:118:11",
          "propertyId": "11",
          "humanReadable": "<He> <PLACE_OF_RESIDENCE> <Brünn>",
          "annotatedPassage": "<He> settled at <Brünn>, in Moravia, and lived about 1756.",
          "subjectStart": 99,
          "subjectEnd": 101,
          "subjectText": "He",
          "subjectUri": "http://www.wikidata.org/entity/Q19517888",
          "objectStart": 113,
          "objectEnd": 118,
          "objectText": "Brünn",
          "objectUri": "http://www.wikidata.org/entity/Q14960"
        },
        {
          "factId": "8313:99:101:123:130:11",
          "propertyId": "11",
          "humanReadable": "<He> <PLACE_OF_RESIDENCE> <Moravia>",
          "annotatedPassage": "<He> settled at Brünn, in <Moravia>, and lived about 1756.",
          "subjectStart": 99,
          "subjectEnd": 101,
          "subjectText": "He",
          "subjectUri": "http://www.wikidata.org/entity/Q19517888",
          "objectStart": 123,
          "objectEnd": 130,
          "objectText": "Moravia",
          "objectUri": "http://www.wikidata.org/entity/Q43266"
        }
      ]
    }
  ]
}
```


### Evaluation
The official evaluation script is also available for download and can be used to evaluate a system using the training set. To use the script, a system has to produce a prediction file, which should contain the same JSON documents of the ground truth file. The prediction file should look exactly like the ground truth file, except for the contents of `facts` (which should contain the facts predicted by the system).

The script has several option parameters (see below), but only the evaluation method (-e) is strictly necessary to execute the script. 

```
usage: evaluator.py [-h] [-e {span_e,span_o,uri}] [-c] [-a ANALYSISPATH] [-f {1,2,3,4,5}]
                    goldFile predictionFile

positional arguments:
  goldFile                Path of the KnowledgeNet file with the gold data
  predictionFile          Path of the KnowledgeNet file with the predicted data

optional arguments:
  -h, --help              show this help message and exit
  -e {span_e,span_o,uri}  Choose the evaluation method: span-exact vs span-overlap vs uri          
  -c                      print raw counts of tp/fn/fp for prec/rec/F1 metrics
  -a ANALYSISPATH         Folder to store error analysis files (default=no analysis).
  -f {1,2,3,4,5}          folds to evaluate (useful during cross-validation). Default is 4.

```

The prediction file has to keep the same unique identifiers and attributes for the given documents and passages. 
Each new fact has to be described by a `factId` (obtained as explained above) and should contain the following attributes that are needed to run the evaluation script: 
* `subjectStart`
* `subjectEnd`
* `objectStart`
* `objectEnd`
* `subjectUri`
* `objectUri`
* `propertyId`

#### Evaluation Methods
Two facts are considered the same when they have the same property, and there is a match between the values for subject and object.

We consider three different methods to establish if there is a match:
* **Span Overlap** (`span_o`): there is an overlap between the character offsets
* **Span Exact** (`span_e`): the character offsets are exactly the same
* **URI** (`uri`): Wikidata URIs are the same (only applies to facts that have URIs for both the subject and the object)

## Adding a system to the leaderboard

To preserve the integrity of the results, we have released the test set (fifth fold) without annotations (`test-no-facts.json`). To evaluate the results of your system and (optionally) add your system to the leaderboard, please send an email with your prediction file to filipe[at]diffbot[dot]com. 