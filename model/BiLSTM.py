#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Tagging 
@File     : BiLSTM.py
@Time     : 2018/2/28 18:06
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""


import tensorflow as tf


class BiLSTM(object):
    """
    A BiLSTM model for word segmentation
    """

    def __init__(self, sequence_length, tag_class, vocab_size, embedding_dim, num_layers,
                 hidden_dim, learning_rate):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, tag_class], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # Embedding layer 指定在cpu
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name='W')
            self.embeded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

        # RNN construct
        def lstm_cell():
            cell = tf.contrib.rnn.LSTMCell(hidden_dim, reuse=tf.get_variable_scope().reuse)
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
        # end of RNN model

        with tf.name_scope('BiLSTM'):
            # forward and backward multi layer LSTM
            cell_fw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
            cell_bw = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)

            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.embeded_chars, dtype=tf.float32)
            output = tf.reshape(tf.concat([output_fw, output_bw], axis=-1), [-1, self.hidden_dim * 2])   # concat最高的维度
            output = tf.nn.dropout(output, self.dropout_keep_prob)

        with tf.name_scope('score'):
            # Dense layer, followed a relu activiation layer
            fc = tf.layers.dense(output, self.hidden_dim, name='fc1')
            fc = tf.nn.dropout(fc, keep_prob=self.dropout_keep_prob)
            fc = tf.nn.relu(fc)

            # classifier
            self.logits = tf.layers.dense(fc, tag_class, name='fc2')
            # probability
            self.prob = tf.nn.softmax(self.logits)
            # prediction
            self.y_pred = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('loss'):
            # loss
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # optimizer
            self.optim = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.y_pred, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))