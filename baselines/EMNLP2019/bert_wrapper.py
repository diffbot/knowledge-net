
from spacy.tokens import Token
import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import FullTokenizer
import numpy as np
import logging

Token.set_extension("bert_vector", default=[])
Token.set_extension("bert_layers", default=[])

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_cased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

LAYERS_TO_USE = 12

bert_outputs_to_use = []
g = tf.Graph()
with g.as_default():
  bert_module = hub.Module(BERT_MODEL_HUB, trainable=False)
  bert_inputs = dict(
    input_ids=tf.placeholder(dtype=tf.int32, shape=[None,None], name="input_ids"),
    input_mask=tf.placeholder(dtype=tf.int32, shape=[None,None], name="input_mask"),
    segment_ids=tf.placeholder(dtype=tf.int32, shape=[None,None], name="segment_ids"))
  bert_outputs = bert_module(bert_inputs, signature="tokens", as_dict=True)
  bert_sequence_output = bert_outputs["sequence_output"]
  def get_intermediate_layer(total_layers, desired_layer):
    intermediate_layer_name = bert_sequence_output.name.replace(str(total_layers + 1),
                                                      str(desired_layer + 1))
    logging.debug("Intermediate layer name: %s", intermediate_layer_name)
    return g.get_tensor_by_name(intermediate_layer_name)
  for i in range(LAYERS_TO_USE):
    bert_outputs_to_use.append(get_intermediate_layer(12, 12-i))
  init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
g.finalize()

sess = tf.compat.v1.Session(graph=g)
sess.run(init_op)

def run(doc):

  max_seq_length = 2

  batch_input_ids = []
  batch_input_mask = []
  batch_segment_ids = []
  batch_orig_to_tok_map = []

  for sentence_index, sentence in enumerate(doc.sents):
    bert_tokens = []
    orig_to_tok_map = []
    segment_ids = [] # sentence index, 1 sentence per so always 0 right now

    bert_tokens.append("[CLS]")
    segment_ids.append(0)
    for word in sentence:
      orig_to_tok_map.append(len(bert_tokens))
      ts = tokenizer.tokenize(word.text)
      if len(ts) == 0:
        logging.debug("Token has no bert tokens: %s", word.text)
      for t in ts:
        bert_tokens.append(t)
        segment_ids.append(0)
    bert_tokens.append("[SEP]")
    segment_ids.append(0)
    orig_to_tok_map.append(len(bert_tokens))

    if len(bert_tokens) > max_seq_length:
      max_seq_length = len(bert_tokens)

    input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    input_mask = [1] * len(input_ids)

    batch_input_ids.append(input_ids)
    batch_input_mask.append(input_mask)
    batch_orig_to_tok_map.append(orig_to_tok_map)
    batch_segment_ids.append(segment_ids)

  # Zero-pad up to the max sequence length.
  for sentence_index, sentence in enumerate(doc.sents):
    while len(batch_input_ids[sentence_index]) < max_seq_length:
        batch_input_ids[sentence_index].append(0)
        batch_input_mask[sentence_index].append(0)
        batch_segment_ids[sentence_index].append(0)

  outputs = sess.run(bert_outputs_to_use, feed_dict={
    bert_inputs["input_ids"]: batch_input_ids,
    bert_inputs["input_mask"]: batch_input_mask,
    bert_inputs["segment_ids"]: batch_segment_ids
  })

  # translate back from subtokens to spacy tokens and store vectors
  for sentence_index, sentence in enumerate(doc.sents):
    for word_index, word in enumerate(sentence):
      bert_vecs = []
      for output in outputs:
        vecs = output[sentence_index][batch_orig_to_tok_map[sentence_index][word_index]:batch_orig_to_tok_map[sentence_index][word_index+1]]
        if len(vecs) == 0:
          #print("Error no output for word", len(output[sentence_index]), batch_orig_to_tok_map[sentence_index][word_index], batch_orig_to_tok_map[sentence_index][word_index+1])
          bert_vecs.append([0]*768)
        else:
          bert_vecs.append(np.average(vecs, axis=0))
      word._.bert_vector = np.concatenate(bert_vecs)
      word._.bert_layers = bert_vecs

# import spacy
# nlp = spacy.load('en_core_web_sm')
# doc = nlp("Mike Tung's the CEO of Diffbot.")
# run(doc)

# for word in doc:
#   print(word._.bert_vector)