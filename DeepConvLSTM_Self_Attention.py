import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Conv2D, Lambda, Input
from tensorflow.keras import backend as k
import Attention_Layers as Al


# Number of unique CAN id's in Dataset
NUM_UNIQUE_CAN_IDS = 23

# Number of classes to predict (Passive (class_label=0) or Aggressive (class_label=1))
NUM_CLASSES = 2

# Sliding window length that segments the data
SLIDING_WINDOW_LENGTH = 60

# Sliding window step that segments the data
# TODO: Sliding window
SLIDING_WINDOW_STEP = 6

# Batch Size
BATCH_SIZE = 16

# Number of filters in the convolutional layers
NUM_FILTERS = 3

# Size of the filters in the convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 32

# Number of hidden units to decode the attention before the softmax activation and becoming annotation weights
ATTENTION_LAYER_SIZE = 32

# Number of hops of attention, or number of distinct components to be extracted from each sentence
ATTENTION_LAYER_HOP_NUM = 10

EPOCH = 5
PATIENCE = 20
SEED = 0
DATA_PATH = os.path.join('Dataset', 'Processed', 'dataset.csv')
SAVE_DIR = 'Saved_Models'

# TODO: Add dropout
def deep_conv_lstm_self_attention():
    input_layer = Input(shape=(1, SLIDING_WINDOW_LENGTH, NUM_UNIQUE_CAN_IDS), batch_size=BATCH_SIZE)

    cnn_1 = Conv2D(NUM_FILTERS, kernel_size=(1, FILTER_SIZE))(input_layer)

    cnn_2 = Conv2D(NUM_FILTERS, kernel_size=(1, FILTER_SIZE))(cnn_1)

    cnn_3 = Conv2D(NUM_FILTERS, kernel_size=(1, FILTER_SIZE))(cnn_2)

    cnn_4 = Conv2D(NUM_FILTERS, kernel_size=(1, FILTER_SIZE))(cnn_3)

    squeeze_layer = Lambda(lambda x: k.squeeze(x, axis=1))(cnn_4)

    # TODO: Dropout and Activation=tanh
    rnn, _, _ = LSTM(NUM_UNITS_LSTM, return_sequences=True, return_state=True)(squeeze_layer)

    attention_layer, _ = Al.SelfAttention(size=ATTENTION_LAYER_SIZE,
                                          num_hops=ATTENTION_LAYER_HOP_NUM,
                                          use_penalization=False,
                                          batch_size=BATCH_SIZE)(rnn)

    dense_layer = Dense(NUM_CLASSES, activation='softmax')(attention_layer)

    dclsa = Model(inputs=input_layer, outputs=dense_layer)
    print(dclsa.summary())
    return dclsa


if __name__ == '__main__':
    print("\nTensorflow version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available!" if tf.config.experimental.list_physical_devices('GPU') else "not available!\n")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if not os.path.exists(os.path.join(SAVE_DIR)):
        os.mkdir(os.path.join(SAVE_DIR))

    # Set random seeds for libraries
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Read in processed dataset
    df = pd.read_csv(DATA_PATH, index_col=None, header=0)

    avg_acc = []
    avg_recall = []
    avg_f1 = []
    early_stopping_epoch_list = []
    X_train = ''
    y_train_one_hot = ''
    X_train_ = ''
    X_test_ = ''
    y_test_one_hot = ''
    y_test = ''

    # Initialise our DeepConvLSTM with Self-Attention Mechanism model
    model = deep_conv_lstm_self_attention()
