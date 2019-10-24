"""Define inputters reading from TFRecord files."""

import tensorflow as tf

from opennmt.inputters.inputter import Inputter
from opennmt.utils import compat
import numpy as np
from collections import defaultdict
import yaml

class Feature:
    def __init__(self, name, shape, where):
        self.name = name
        self.shape = shape
        self.where = where

class RecordInputter(Inputter):
  """Inputter that reads a header file that discribes the tensors and shapes
  """

  def __init__(self, dtype=tf.float32):
    """Initializes the parameters of the record inputter.

    Args:
      dtype: The values type.
    """
    super(RecordInputter, self).__init__(dtype=dtype)

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    config_file = metadata['config_file']
    # read config file
    self.input_features = []
    self.has = defaultdict(bool)
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
        for fi in config["features"]:
            f = Feature(fi[0], fi[1], fi[2])
            self.has[f.where] = True
            self.input_features.append(f)
    print(self.input_features)

  def make_dataset(self, data_file, training=None):
    return tf.data.TFRecordDataset(data_file)

  def get_dataset_size(self, data_file):
    return sum(1 for _ in compat.tf_compat(v1="python_io.tf_record_iterator")(data_file))

  def get_receiver_tensors(self):
    ret = {}
    if self.has_word():
        ret["numWords"] = tf.placeholder(tf.int32, shape=(None,), name="numWords")
    for feature in self.input_features:
        shape = list(map( lambda x: None if x < 0 else x, list(feature.shape)))
        shape.insert(0, None) # batch size
        ret[feature.name] = tf.placeholder(tf.float32, shape=tuple(shape), name=feature.name)
    return ret

  def make_features(self, element=None, features=None, training=None):
    if features is None:
        features = {}
    if self.input_features[0].name in features:
        return features
    if element is None:
        raise RuntimeError("make_features was called with None element")
    tf_parse_example = compat.tf_compat(v2="io.parse_single_example", v1="parse_single_example")
    tf_var_len_feature = compat.tf_compat(v2="io.VarLenFeature", v1="VarLenFeature")
    featuresDict = {}
    if self.has_word():
        featuresDict["numWords"] = tf_var_len_feature(tf.int64)
    
    for feature in self.input_features:
        featuresDict[feature.name] = tf_var_len_feature(tf.float32)

    example = tf_parse_example(element, features=featuresDict)

    if self.has_word():
      features["numWords"] = tf.cast(example["numWords"].values, tf.int32)[0]

    for feature in self.input_features:
        print(feature.name, feature.shape)
        features[feature.name] = tf.reshape(example[feature.name].values, feature.shape)

    print("features", features)
    return features

  
  def get_word(self, features, training=None):
    to_concat = []
    for feature in self.input_features:
        if feature.where == "word":
            to_concat.append(features[feature.name])
    return tf.concat(to_concat, axis=-1)

  def get_global(self, features, training=None):
    to_concat = []
    for feature in self.input_features:
        if feature.where == "global":
            to_concat.append(features[feature.name])
    if len(to_concat) == 1:
      return to_concat[0]
    return tf.concat(to_concat, axis=-1)

  def get_lm(self, features, training=None):
    to_concat = []
    for feature in self.input_features:
        if feature.where == "lm":
            to_concat.append(features[feature.name])
    return tf.concat(to_concat, axis=-2)

  def has_word(self):
    return self.has["word"]
  
  def has_lm(self):
    return self.has["lm"]

  def has_global(self):
    return self.has["global"]

  def get_length(self, features, training=None):
    return features["numWords"] if "numWords" in features else 1