## Requirements
python 3.7
```
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

Set environment variable DIFFBOT_TOKEN if you want to use entitylinking. We are providing cached results for the KnowledgeNet documents, but you will need this if you want to run the system on other documents or if you want to change the NER system. Free trial available at https://www.diffbot.com/get-started/

## Using the pretrained model

Get the release from https://github.com/diffbot/knowledge-net/releases which includes the pretrained baseline 5 model, vocab, and linking and wikidata cache.

Run on a single document:

`echo "Butler W. Lampson (born December 23, 1943) is an American computer scientist contributing to the development and implementation of distributed, personal computing. He is a Technical Fellow at Microsoft and an Adjunct Professor at MIT." | python run.py`

## Evaluating

python evaluate.py [test or dev]

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
