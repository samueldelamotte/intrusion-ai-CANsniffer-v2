import os
import random
import numpy as np
import itertools as it
import tensorflow as tf
import sklearn as skl
from keras_self_attention import SeqSelfAttention
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input

# Number of classes to predict (Passive (class_label=0) or Aggressive (class_label=1))
NUM_CLASSES = 2
# Sliding window length that segments the data
SLIDING_WINDOW_LENGTH = 60
# Sliding window step that segments the data
SLIDING_WINDOW_STEP = 6
# Batch Size
BATCH_SIZE = 128
# Number of filters in the convolutional layers
NUM_FILTERS = 64
# Size of the filters in the convolutional layers
FILTER_SIZE = 3
# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 60
ATTENTION_UNITS = 60
EPOCHS = 25
SEED = 0
PASSIVE_DATA_PATH = os.path.join('Dataset', 'Processed', 'Passive', 'passive-dataset.csv')
NPY_PASSIVE_DATA_PATH = os.path.join('Dataset', 'Processed', 'Passive', 'passive-dataset.npy')
AGGRESSIVE_DATA_PATH = os.path.join('Dataset', 'Processed', 'Aggressive', 'aggressive-dataset.csv')
NPY_AGGRESSIVE_DATA_PATH = os.path.join('Dataset', 'Processed', 'Aggressive', 'aggressive-dataset.npy')
MODEL_SAVE_DIR = os.path.join('Saved_Models')


def model():
    # Input shape is num of time steps * features
    input_layer = Input(shape=(60, 10), batch_size=BATCH_SIZE)
    conv = Conv1D(NUM_FILTERS, 1, padding='same', activation='relu', input_shape=(60, 10))(input_layer)
    conv2 = Conv1D(NUM_FILTERS, 1, padding='same', activation='relu')(conv)
    conv3 = Conv1D(NUM_FILTERS, 1, padding='same', activation='relu')(conv2)
    lstm = LSTM(NUM_UNITS_LSTM, dropout=0.25, activation='tanh', return_sequences=True)(conv3)
    attention = SeqSelfAttention(ATTENTION_UNITS, attention_width=60, attention_activation='tanh')
    attention_out = attention(lstm)
    dense = Dense(1024, activation='relu')(attention_out)
    output_layer = Dense(1, activation='sigmoid')(dense)
    model_ = Model(inputs=input_layer, outputs=output_layer)
    print(model_.summary())
    return model_


def sliding_window(data, length, step=1):
    streams = it.tee(data, length)
    a = zip(*[it.islice(stream, i, None, step*length) for stream, i in zip(streams, it.count(step=step))])
    b = list(a)
    return np.asarray(b)


if __name__ == '__main__':
    print("\nTensorflow version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available!" if tf.config.experimental.list_physical_devices('GPU') else "not available!\n")
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Check if model save directory exists, else mkdir
    if not os.path.exists(os.path.join(MODEL_SAVE_DIR)):
        os.mkdir(os.path.join(MODEL_SAVE_DIR))

    # Set random seeds for libraries
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    passive = []
    aggressive = []

    # Check if passive numpy saves exist, if so load .npy, else read in .csv
    if os.path.exists(os.path.join(NPY_PASSIVE_DATA_PATH)):
        passive = np.load(NPY_PASSIVE_DATA_PATH)
    else:
        passive = np.genfromtxt(PASSIVE_DATA_PATH, delimiter=',', skip_header=1, dtype=int)
        np.save(NPY_PASSIVE_DATA_PATH, passive)

    # Check if aggressive numpy saves exist, if so load .npy, else read in .csv
    if os.path.exists(os.path.join(NPY_AGGRESSIVE_DATA_PATH)):
        aggressive = np.load(NPY_AGGRESSIVE_DATA_PATH)
    else:
        aggressive = np.genfromtxt(AGGRESSIVE_DATA_PATH, delimiter=',', skip_header=1, dtype=int)
        np.save(NPY_AGGRESSIVE_DATA_PATH, aggressive)

    # Extract labels
    passive_labels = passive[:, -1]
    aggressive_labels = aggressive[:, -1]

    # Generate sliding windows on passive and aggressive
    passive_ = sliding_window(passive[:, :-1], SLIDING_WINDOW_LENGTH)
    passive_labels_ = sliding_window(passive_labels, SLIDING_WINDOW_LENGTH)
    aggressive_ = sliding_window(aggressive[:, :-1], SLIDING_WINDOW_LENGTH)
    aggressive_labels_ = sliding_window(aggressive_labels, SLIDING_WINDOW_LENGTH)

    # Concatenate and then shuffle the dataset windows around
    X = np.concatenate((passive_, aggressive_))
    y = np.concatenate((passive_labels_, aggressive_labels_))
    X_, y_ = skl.utils.shuffle(X, y, random_state=SEED)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.2, random_state=SEED)

    # Initialise our DeepConvLSTM with Self-Attention Mechanism model
    model = model()

    # Print the shapes and number of training instances and testing instances
    print('Number of training instances:', X_train.shape[0])
    print('Number of test instances:    ', X_test.shape[0])

    # X_train, y_train, X_test, and y_test are all numpy arrays
    print('X_train.shape =', X_train.shape)
    print('y_train.shape =', y_train.shape)
    print('X_test.shape =', X_test.shape)
    print('y_test.shape =', y_test.shape)

    # Fit the model to the training data
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.05)
    y_pred_score = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)

    # Let's take a look at our model's accuracy score
    print("DeepConvLSTM-Self-Attention accuracy: ", y_pred_score[1])
    # Let's take a look at our model's loss
    print("DeepConvLSTM-Self-Attention loss: ", y_pred_score[0])
