import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Flatten, Activation
from tensorflow.keras.layers import Multiply, Lambda, Reshape, Dot, Concatenate, RepeatVector, TimeDistributed, Permute, Bidirectional

"""
Created on Wed Jun 19 20:08:11 2019

@author: ongunuzaymacar

Script containing custom layer implementations for a family of attention mechanisms in TensorFlow
with Keras integration (tested for TF 2.0). Comments next to each operation in each layer indicate
the output shapes. For ease of notation, the following abbreviations are used:
i)    B  = batch size,
ii)   S  = sequence length (many-to-one) OR input sequence length (many-to-many),
iii)  S' = target sequence length (many-to-many),
iv)   S* = optimized (by 'local' approach, sometimes referred to as 'alignment length') sequence
           length,
v)    S- = the larger of the sequence lengths for many-to-many scenarios,
vi)   V  = vocabulary size,
vii)  H  = number of hidden dimensions,
viii) E  = number of embedding dimensions

Additionally, if a tensors shape differs for many-to-one and many-to-many scenarios, <1> and <M>
tags will respectively identify the corresponding shapes. If no distinction is made, assume that
the shape indicated is applicable for both scenarios.
"""

class SelfAttention(Layer):
    """
    Layer for implementing self-attention mechanism. Weight variables were preferred over Dense()
    layers in implementation because they allow easier identification of shapes. Softmax activation
    ensures that all weights sum up to 1.

    @param (int) size: a.k.a attention length, number of hidden units to decode the attention before
           the softmax activation and becoming annotation weights
    @param (int) num_hops: number of hops of attention, or number of distinct components to be
           extracted from each sentence.
    @param (bool) use_penalization: set True to use penalization, otherwise set False
    @param (int) penalty_coefficient: the weight of the extra loss
    @param (str) model_api: specify to use TF's Sequential OR Functional API, note that attention
           weights are not outputted with the former as it only accepts single-output layers
    """
    def __init__(self, size, num_hops=8, use_penalization=True,
                 penalty_coefficient=0.1, model_api='functional', batch_size = 1, **kwargs):
        if model_api not in ['sequential', 'functional']:
            raise ValueError("Argument for param @model_api is not recognized")
        self.size = size
        self.num_hops = num_hops
        self.use_penalization = use_penalization
        self.penalty_coefficient = penalty_coefficient
        self.model_api = model_api
        self.batch_size = batch_size
        super(SelfAttention, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(SelfAttention, self).get_config()
        base_config['size'] = self.size
        base_config['batch_size'] = self.batch_size
        base_config['num_hops'] = self.num_hops
        base_config['use_penalization'] = self.use_penalization
        base_config['penalty_coefficient'] = self.penalty_coefficient
        base_config['model_api'] = self.model_api
        return base_config

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.size, int(input_shape[2])),                                # (size, H)
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.num_hops, self.size),                                 # (num_hops, size)
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):  # (B, S, H)
        # Expand weights to include batch size through implicit broadcasting
        W1, W2 = self.W1[None, :, :], self.W2[None, :, :]
        W1, W2 = tf.tile(W1, [self.batch_size, 1, 1]), tf.tile(W2, [self.batch_size, 1, 1])
        #W1, W2 = tf.compat.v1.repeat(W1, repeats = [self.batch_size], axis=0), tf.compat.v1.repeat(W2, repeats = [self.batch_size], axis=0)
        hidden_states_transposed = Permute(dims=(2, 1))(inputs)                                     # (B, H, S)
        attention_score = tf.matmul(W1, hidden_states_transposed)                                   # (B, size, S)
        attention_score = Activation('tanh')(attention_score)                                       # (B, size, S)
        attention_weights = tf.matmul(W2, attention_score)                                          # (B, num_hops, S)
        attention_weights = Activation('softmax')(attention_weights)                                # (B, num_hops, S)
        embedding_matrix = tf.matmul(attention_weights, inputs)                                     # (B, num_hops, H)
        embedding_matrix_flattened = Flatten()(embedding_matrix)                                    # (B, num_hops*H)

        if self.use_penalization:
            attention_weights_transposed = Permute(dims=(2, 1))(attention_weights)                  # (B, S, num_hops)
            product = tf.matmul(attention_weights, attention_weights_transposed)                    # (B, num_hops, num_hops)
            identity = tf.eye(self.num_hops, batch_shape=(inputs.shape[0],))                        # (B, num_hops, num_hops)
            frobenius_norm = tf.sqrt(tf.reduce_sum(tf.square(product - identity)))  # distance
            self.add_loss(self.penalty_coefficient * frobenius_norm)  # loss

        if self.model_api == 'functional':
            return embedding_matrix_flattened, attention_weights
        elif self.model_api == 'sequential':
            return embedding_matrix_flattened
