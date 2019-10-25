"""Sequence classifier."""

import tensorflow as tf

from opennmt import inputters, decoders, layers
from opennmt.models.model import Model
from opennmt.utils.cell import last_encoding_from_state
from opennmt.utils.misc import print_bytes
from opennmt.utils.losses import cross_entropy_loss
from opennmt.layers.reducer import ConcatReducer
from opennmt.utils import compat
from opennmt.utils.misc import count_lines
import yaml
import numpy as np

class PositionEmbedder(tf.keras.layers.Layer):
  """Encodes position with a lookup table."""

  def __init__(self, maximum_position=32, reducer=ConcatReducer()):
    """Initializes the position encoder.
    Args:
      maximum_position: The maximum position to embed. Positions greater
        than this value will be set to :obj:`maximum_position`.
      reducer: A :class:`opennmt.layers.reducer.Reducer` to merge inputs and
        position encodings.
    """
    super(PositionEmbedder, self).__init__()
    self.maximum_position = maximum_position
    self.embedding = None

  def build(self, i, dtype=tf.float32):
    shape = [self.maximum_position+1, 4]
    initializer = tf.keras.initializers.glorot_uniform()
    self.embedding = tf.Variable(
        initial_value=lambda: initializer(shape, dtype=dtype),
        name=compat.name_from_variable_scope("position_encoding/w_embs/"+str(i)))

  def encode(self, positions):
    positions = tf.minimum(positions, self.maximum_position)
    #positions = tf.maximum(positions, -1*self.maximum_position)
    #c = tf.constant(self.maximum_position)
    #indexes = tf.math.add(positions, c)
    return tf.nn.embedding_lookup(self.embedding, positions)

class SequenceClassifier(Model):
  """A sequence classifier."""

  def __init__(self,
               inputter,
               encoder,
               labels_vocabulary_file_key,
               encoding="average",
               daisy_chain_variables=False,
               name="seqclassifier"):
    """Initializes a sequence classifier.

    Args:
      inputter: A :class:`opennmt.inputters.inputter.Inputter` to process the
        input data.
      encoder: A :class:`opennmt.encoders.encoder.Encoder` to encode the input.
      labels_vocabulary_file_key: The data configuration key of the labels
        vocabulary file containing one label per line.
      encoding: "average" or "last" (case insensitive), the encoding vector to
        extract from the encoder outputs.
      daisy_chain_variables: If ``True``, copy variables in a daisy chain
        between devices for this model. Not compatible with RNN based models.
      name: The name of this model.

    Raises:
      ValueError: if :obj:`encoding` is invalid.
    """
    super(SequenceClassifier, self).__init__(
        name,
        features_inputter=inputter,
        labels_inputter=LabelsInputter(),
        daisy_chain_variables=daisy_chain_variables)
    self.encoder = encoder
    self.encoding = encoding.lower()
    if self.encoding not in ("average", "last"):
      raise ValueError("Invalid encoding vector: {}".format(self.encoding))
    self.position_encoders = []
    #for i in range(inputter.numToEmbed):
    #  self.position_encoders.append(PositionEmbedder())

  def initialize(self, metadata):
    """Initializes the model from the data configuration.
    Args:
      metadata: A dictionary containing additional data configuration set
        by the user (e.g. vocabularies, tokenization, pretrained embeddings,
        etc.).
    """
    super(SequenceClassifier, self).initialize(metadata)
    config_file = metadata['config_file']
    # read config file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        self.number_of_labels = len(config["predicates"]) * config["outputs_per_predicate"]
    print(self.number_of_labels)
    self.labels_inputter.number_of_outputs(self.number_of_labels)

  def _call(self, features, labels, params, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN
    if self.features_inputter.has_word():
      with tf.variable_scope("encoder"):
        inputs = self.features_inputter.get_word(features, training=training)

        to_concat = [ inputs ]

        if self.features_inputter.has_lm():
          lm_layers = self.features_inputter.get_lm(features)
          lm_layer_weights = tf.nn.softmax(tf.Variable(tf.random.uniform([lm_layers.shape[-2].value], maxval=1)), axis=0)
          w = tf.tensordot(lm_layers, lm_layer_weights, axes=[[2],[0]])
          to_concat.append(w)

        inputs = tf.concat(to_concat, axis=-1)

        encoder_outputs, encoder_state, _ = self.encoder.encode(
            inputs,
            sequence_length=self.features_inputter.get_length(features),
            mode=mode)

      if self.encoding == "average":
        encoding = tf.reduce_mean(encoder_outputs, axis=1)
      elif self.encoding == "last":
        encoding = last_encoding_from_state(encoder_state)

    if self.features_inputter.has_global():
      global_features = self.features_inputter.get_global(features)
      print(global_features)

      encoding = tf.concat([encoding, global_features], axis=-1) if self.features_inputter.has_word() else global_features
      encoding = tf.layers.dense(encoding, 256, activation='relu')

    with tf.variable_scope("generator"):
      logits = tf.layers.dense(
          encoding,
          self.number_of_labels)

    if mode != tf.estimator.ModeKeys.TRAIN:
      prob = tf.math.sigmoid(logits)
      predictions = {
          "probabilities": prob,
          "labels": tf.round(prob)
      }
    else:
      predictions = None

    return logits, predictions

  def compute_loss(self, outputs, labels, training=True, params=None):
    if params is None:
      params = {}
    labels_flat = tf.reshape(labels["labels"], shape=[-1])
    output_flat = tf.reshape(outputs, shape=[-1])
    mask = tf.not_equal(labels_flat, tf.constant(-1.0, dtype=tf.float32)) # labels are 1=positive 0=negative -1=unknown
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits= tf.boolean_mask(output_flat, mask),
        labels= tf.boolean_mask(labels_flat, mask))
    return tf.reduce_sum(loss)

  def compute_metrics(self, predictions, labels):
    labels_flat = tf.reshape(labels["labels"], shape=[-1])
    predictions_flat = tf.reshape(predictions["labels"], shape=[-1])
    mask = tf.not_equal(labels_flat, tf.constant(-1.0, dtype=tf.float32)) # labels are 1=positive 0=negative -1=unknown
    return {
        "accuracy": tf.metrics.accuracy(tf.boolean_mask(labels_flat, mask), tf.boolean_mask(predictions_flat, mask))
    }

  def print_prediction(self, prediction, params=None, stream=None):
    print(','.join(str(x) for x in prediction["probabilities"]), file=stream)

class LabelsInputter(inputters.Inputter):
  """Reading class from a text file."""

  def __init__(self):
    super(LabelsInputter, self).__init__(dtype=tf.float32)

  def number_of_outputs(self, num_outputs):
    self.field_defaults = [[] for _ in range(num_outputs)]

  def _parse_line(self, line):
    return tf.decode_csv(line, self.field_defaults)

  def make_dataset(self, data_file, training=None):
    ds = tf.data.TextLineDataset(data_file)
    print(ds)
    return ds
  def get_dataset_size(self, data_file):
    return count_lines(data_file)

  def make_features(self, element=None, features=None, training=None):
    if features is None:
      features = {}
    if "labels" in features:
      return features
    tokens = self._parse_line(element)
    features["length"] = tf.shape(tokens)[0]
    features["labels"] = tokens
    return features

