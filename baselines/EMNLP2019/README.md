## Requirements
python 3.7
```
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Set environment variable DIFFBOT_TOKEN if you want to use entitylinking. We are providing cached results for the KnowledgeNet documents, but you will need this if you want to run the system on other documents or if you want to change the NER system. Contact Filipe Mesquita (filipe[at]diffbot.com) for a free research token.

## Using the pretrained model

Get the release from https://github.com/diffbot/knowledge-net/releases which includes the pretrained baseline 5 model, vocab, and linking and wikidata cache.

Run on a single document:

`echo "Butler W. Lampson (born December 23, 1943) is an American computer scientist contributing to the development and implementation of distributed, personal computing. He is a Technical Fellow at Microsoft and an Adjunct Professor at MIT." | python run.py`

output
```
Butler W. Lampson| DATE_OF_BIRTH(0.99) December 23, 1943|
Butler W. Lampson|http://www.wikidata.org/entity/Q92644 NATIONALITY(0.88) American|http://www.wikidata.org/entity/Q30
Microsoft|http://www.wikidata.org/entity/Q2283 CEO(0.63) He|http://www.wikidata.org/entity/Q92644
He|http://www.wikidata.org/entity/Q92644 EMPLOYEE_OR_MEMBER_OF(0.88) Microsoft|http://www.wikidata.org/entity/Q2283
He|http://www.wikidata.org/entity/Q92644 EMPLOYEE_OR_MEMBER_OF(0.93) MIT|http://www.wikidata.org/entity/Q49108
He| DATE_OF_BIRTH(1.00) December 23, 1943|
He|http://www.wikidata.org/entity/Q92644 NATIONALITY(0.86) American|http://www.wikidata.org/entity/Q30
Butler W. Lampson|http://www.wikidata.org/entity/Q92644 EMPLOYEE_OR_MEMBER_OF(0.56) Microsoft|http://www.wikidata.org/entity/Q2283
Butler W. Lampson|http://www.wikidata.org/entity/Q92644 EMPLOYEE_OR_MEMBER_OF(0.69) MIT|http://www.wikidata.org/entity/Q49108
```

## Evaluating

`python evaluate.py [test or dev]`

This creates the analysis files in `tmp` and when run on `dev` prints the results. To preserve the integrity of the results, we have released the test set without annotations. See https://github.com/diffbot/knowledge-net#adding-a-system-to-the-leaderboard for more details.

With the pretrained baseline 5 model you should get similar to the following on the dev set
```
Evaluation     Precision      Recall         F1
span_overlap   0.718          0.691          0.704
span_exact     0.620          0.599          0.609
uri            0.557          0.472          0.511  
```

## Training

Choose which model you would like to train in config.py

Warning: baseline 5 requires ~300GB of disk space to train. The others require much less.

`./train.sh`

## Troubleshooting

`spacy.strings.StringStore size changed error`

If you have an error mentioning spacy.strings.StringStore size changed, may indicate binary incompatibility when loading NeuralCoref with import neuralcoref, it means you'll have to install NeuralCoref from the distribution's sources instead of the wheels to get NeuralCoref to build against the most recent version of SpaCy for your system.

In this case, simply re-install neuralcoref as follows:

pip uninstall neuralcoref

pip install neuralcoref --no-binary neuralcoref
