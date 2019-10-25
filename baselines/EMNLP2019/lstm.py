
"""LSTM"""

import tensorflow as tf
import opennmt as onmt
from sequence_classifier import SequenceClassifier
from general_inputter import RecordInputter

def model():
  return SequenceClassifier(
        inputter=RecordInputter(),
        encoder=onmt.encoders.BidirectionalRNNEncoder(
            num_layers=2,
            num_units=500,
            reducer=onmt.layers.ConcatReducer(),
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.0,
            residual_connections=True),
        encoding="average",
        labels_vocabulary_file_key="labels_vocabulary")
