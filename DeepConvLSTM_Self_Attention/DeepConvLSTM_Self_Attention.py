import os
import random
import numpy as np
import pandas as pd
import scipy.stats as st
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, LSTM, Conv2D, Lambda, Input, InputLayer, MaxPooling2D, Permute
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from sklearn.metrics import f1_score, accuracy_score, recall_score, mean_absolute_error
import Attention_Layers as al


# def deep_conv_lstm_self_attention(x_train, num_labels, LSTM_units, num_conv_filters, batch_size, f, d):
#     """
#     The proposed model with CNN layer, LSTM RNN layer and self attention layers.
#     Inputs:
#     - x_train: required for creating input shape for RNN layer in Keras
#     - num_labels: number of output classes (int)
#     - LSTM_units: number of RNN units (int)
#     - num_conv_filters: number of CNN filters (int)
#     - batch_size: number of samples to be processed in each batch
#     - F: the attention length (int)
#     - D: the length of the output (int)
#     Returns
#     - model: A Keras model
#     """
#     cnn_inputs_1 = Input(shape=(x_train.shape[1], x_train.shape[2], 1), batch_size=batch_size, name='CNN')
#     cnn_layer_1 = Conv2D(num_conv_filters, kernel_size = (1, x_train.shape[2]), strides=(1, 1), padding='valid', data_format="channels_last")
#     cnn_out_1 = cnn_layer_1(cnn_inputs_1)

#     pooling_layer = MaxPooling2D()
#     pooling_out = pooling_layer(cnn_out_1)

#     cnn_layer_2 = Conv2D(num_conv_filters, kernel_size = (1, x_train.shape[2]), strides=(1, 1), padding='valid', data_format="channels_last")
#     cnn_out_2 = cnn_layer_2(pooling_out)

#     sq_layer = Lambda(lambda x: K.squeeze(x, axis = 2))
#     sq_layer_out = sq_layer(cnn_out_2)

#     rnn_layer = LSTM(LSTM_units, recurrent_dropout=0.5, activation='relu', return_sequences=True, name='RNN', return_state=True) #return_state=True
#     rnn_layer_output, _, _ = rnn_layer(sq_layer_out)

#     encoder_output, _ = al.SelfAttention(size=f, num_hops=d, use_penalization=False, batch_size = batch_size)(rnn_layer_output)
#     dense_layer = Dense(num_labels, activation = 'softmax')
#     dense_layer_output = dense_layer(encoder_output)

#     model = Model(inputs=cnn_inputs_1, outputs=dense_layer_output)
#     print (model.summary())

#     return model


# Number of unique CAN id's in Dataset
NUM_UNIQUE_CAN_IDS = 23

# Number of classes to predict (Aggresive or Passive)
NUM_CLASSES = 2

# Sliding window length that segments the data
SLIDING_WINDOW_LENGTH = 24

# Sliding window step that segments the data
SLIDING_WINDOW_STEP = 12

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Batch Size
BATCH_SIZE = 100

# Number of filters in the convolutional layers
NUM_FILTERS = 64

# Size of the filters in the convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

# 
ATTENTION_LAYER_SIZE = 32

# 
ATTENTION_LAYER_HOP_NUM = 10

EPOCH = 1
PATIENCE = 20
SEED = 0
DATA_FILES = ['WISDM.npz']
MODE = 'LOTO'
BASE_DIR = 'Dataset/' + MODE + '/'
SAVE_DIR = 'Dataset/' + MODE + '/Saved_Models/'

"""
    net = {}
    net['input'] = lasagne.layers.InputLayer((BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
    net['conv1/5x1'] = lasagne.layers.Conv2DLayer(net['input'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['conv2/5x1'] = lasagne.layers.Conv2DLayer(net['conv1/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['conv3/5x1'] = lasagne.layers.Conv2DLayer(net['conv2/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['conv4/5x1'] = lasagne.layers.Conv2DLayer(net['conv3/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
    net['shuff'] = lasagne.layers.DimshuffleLayer(net['conv4/5x1'], (0, 2, 1, 3))
    net['lstm1'] = lasagne.layers.LSTMLayer(net['shuff'], NUM_UNITS_LSTM)
    net['lstm2'] = lasagne.layers.LSTMLayer(net['lstm1'], NUM_UNITS_LSTM)
    # In order to connect a recurrent layer to a dense layer, it is necessary to flatten the first two dimensions
    # to cause each time step of each sequence to be processed independently (see Lasagne docs for further information)
    net['shp1'] = lasagne.layers.ReshapeLayer(net['lstm2'], (-1, NUM_UNITS_LSTM))
    net['prob'] = lasagne.layers.DenseLayer(net['shp1'],NUM_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
    # Tensors reshaped back to the original shape
    net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'], (BATCH_SIZE, FINAL_SEQUENCE_LENGTH, NUM_CLASSES))
    # Last sample in the sequence is considered
    net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)
"""

def deepConvLstmSelfAttention():
    input_layer = InputLayer(input_shape=(BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NUM_UNIQUE_CAN_IDS))
    cnn_1_in = Conv2D(NUM_FILTERS, kernel_size=(FILTER_SIZE, 1))
    cnn_1_out = cnn_1_in(input_layer)

    cnn_2_in = Conv2D(NUM_FILTERS, kernel_size=(FILTER_SIZE, 1))
    cnn_2_out = cnn_2_in(cnn_1_out)

    cnn_3_in = Conv2D(NUM_FILTERS, kernel_size=(FILTER_SIZE, 1))
    cnn_3_out = cnn_3_in(cnn_2_out)

    cnn_4_in = Conv2D(NUM_FILTERS, kernel_size=(FILTER_SIZE, 1))
    cnn_4_out = cnn_4_in(cnn_3_out)

    permute_layer_in = Permute((0, 2, 1, 3))
    permute_layer_out = permute_layer_in(cnn_4_out)

    rnn_1_in = LSTM(NUM_UNITS_LSTM)
    rnn_1_out = rnn_1_in(permute_layer_out)

    rnn_2_in = LSTM(NUM_UNITS_LSTM)
    rnn_2_out = rnn_2_in(rnn_1_out)

    attention_layer, _ = al.SelfAttention(size=ATTENTION_LAYER_SIZE, num_hops=ATTENTION_LAYER_HOP_NUM, use_penalization=False, batch_size=BATCH_SIZE)(rnn_2_out)

    dense_layer_in = Dense(NUM_CLASSES, activation = 'softmax')
    dense_layer_out = dense_layer_in(attention_layer)

    dclsa = Model(inputs=input_layer, outputs=dense_layer_out)
    print (dclsa.summary())
    return dclsa

if __name__ == '__main__':
    print("\nTensorflow version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print("GPU is", "available!" if tf.config.experimental.list_physical_devices('GPU') else "not available!\n")

    # session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]= '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # K.set_session(sess)

    if not os.path.exists(os.path.join(SAVE_DIR)):
        os.mkdir(os.path.join(SAVE_DIR))

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(0)

    for DATA_FILE in DATA_FILES:
        data_input_file = os.path.join(BASE_DIR, DATA_FILE)
        tmp = np.load(data_input_file, allow_pickle=True)
        # df = pd.DataFrame.from_dict({item: tmp[item] for item in tmp.files}, orient='index')
        # print(df.head + '\n')
        X = tmp['X']
        X = np.squeeze(X, axis = 1)
        y_one_hot = tmp['y']
        folds = tmp['folds']

        NUM_LABELS = y_one_hot.shape[1]

        avg_acc = []
        avg_recall = []
        avg_f1 = []
        early_stopping_epoch_list = []
        y = np.argmax(y_one_hot, axis=1)

        for i in range(0, len(folds)):
            train_idx = folds[i][0]
            test_idx = folds[i][1]

            X_train, y_train, y_train_one_hot = X[train_idx], y[train_idx], y_one_hot[train_idx]
            X_test, y_test, y_test_one_hot = X[test_idx], y[test_idx], y_one_hot[test_idx]

            X_train_ = np.expand_dims(X_train, axis = 3)
            X_test_ = np.expand_dims(X_test, axis = 3)

            train_trailing_samples =  X_train_.shape[0]%BATCH_SIZE
            test_trailing_samples =  X_test_.shape[0]%BATCH_SIZE


            if train_trailing_samples!= 0:
                X_train_ = X_train_[0:-train_trailing_samples]
                y_train_one_hot = y_train_one_hot[0:-train_trailing_samples]
                y_train = y_train[0:-train_trailing_samples]
            if test_trailing_samples!= 0:
                X_test_ = X_test_[0:-test_trailing_samples]
                y_test_one_hot = y_test_one_hot[0:-test_trailing_samples]
                y_test = y_test[0:-test_trailing_samples]

            print(y_train.shape, y_test.shape)

            # dclsa_model = deep_conv_lstm_self_attention(x_train = X_train_, num_labels = NUM_LABELS, LSTM_units = LSTM_UNITS, \
            #     num_conv_filters = CNN_FILTERS, batch_size = BATCH_SIZE, f = F, d = D)

            dclsa_model = deepConvLstmSelfAttention()

            model_filename = SAVE_DIR + 'best_model_with_self_attn_' + str(DATA_FILE[0:-4]) + '_fold_' + str(i) + '.h5'
            callbacks = [ModelCheckpoint(filepath=model_filename, monitor = 'val_accuracy', save_weights_only=True, save_best_only=True), EarlyStopping(monitor='val_accuracy', patience=PATIENCE)]#, LearningRateScheduler()]

            opt = optimizers.Adam(clipnorm=1.)

            dclsa_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

            history = dclsa_model.fit(X_train_, y_train_one_hot, epochs=EPOCH, batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks, validation_data=(X_test_, y_test_one_hot))

            early_stopping_epoch = callbacks[1].stopped_epoch - PATIENCE + 1
            print('Early stopping epoch: ' + str(early_stopping_epoch))
            early_stopping_epoch_list.append(early_stopping_epoch)

            if early_stopping_epoch <= 0:
                early_stopping_epoch = -100

            # Evaluate model and predict data on TEST
            print("\n******Evaluating TEST set*********\n")
            dclsa_model.load_weights(model_filename)

            y_test_predict = dclsa_model.predict(X_test_, batch_size = BATCH_SIZE)
            y_test_predict = np.array(y_test_predict)
            y_test_predict = np.argmax(y_test_predict, axis=1)

            # all_trainable_count = int(np.sum([K.count_params(p) for p in set(dclsa_model.trainable_weights)]))

            MAE = mean_absolute_error(y_test, y_test_predict, sample_weight=None, multioutput='uniform_average')

            acc_fold = accuracy_score(y_test, y_test_predict)
            avg_acc.append(acc_fold)

            recall_fold = recall_score(y_test, y_test_predict, average='macro')
            avg_recall.append(recall_fold)

            f1_fold  = f1_score(y_test, y_test_predict, average='macro')
            avg_f1.append(f1_fold)

            with open(SAVE_DIR + '/results_model_with_self_attn_' + MODE + '.csv', 'a') as out_stream:
                out_stream.write(str(SEED) + ', ' + str(DATA_FILE[0:-4]) + ', ' + str(i) + ', ' + str(early_stopping_epoch) + ', '  + ', ' + str(acc_fold) + ', ' + str(MAE) + ', ' + str(recall_fold) + ', ' + str(f1_fold) + '\n')


            print('Accuracy[{:.4f}] Recall[{:.4f}] F1[{:.4f}] at fold[{}]'.format(acc_fold, recall_fold, f1_fold, i))
            print('______________________________________________________')
            K.clear_session()

    ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
    ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
    ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc = np.mean(avg_f1), scale=st.sem(avg_f1))

    print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
    print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
    print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))