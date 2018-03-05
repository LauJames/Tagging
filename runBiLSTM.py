#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Tagging 
@File     : runBiLSTM.py
@Time     : 2018/3/2 10:35
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import os
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
import csv
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from model.BiLSTM import BiLSTM
from data import dataHelper

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("test_sample_percentage", 0.2, "Percentage of the training data to use for test")
tf.flags.DEFINE_string("train_data_file", "./data/data.pkl",
                       "Data source for the train data.")
tf.flags.DEFINE_string("tensorboard_dir", "tensorboard_dir/BiLSTM", "saving path of tensorboard")
tf.flags.DEFINE_string("save_dir", "checkpoints/BiLSTM", "save base dir")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("timestep_size", 32, "max_len = timestep_size (default: 32)")
tf.flags.DEFINE_integer("vocab_size", 5159, "vocabulary size (according to data processing)")
tf.flags.DEFINE_integer("num_classes", 5, "Number of classes (default: 5)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_layers", 2, "number of layers (default: 2)")
tf.flags.DEFINE_integer("hidden_dim", 128, "neural numbers of hidden layer (default: 128)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "learning rate (default:1e-3)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
save_path = os.path.join(FLAGS.save_dir, 'best_validation')


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: keep_prob
    }
    return feed_dict


def evaluate(x_dev, y_dev, sess):
    """
    Evaluates model on a dev set
    :param x_dev:
    :param y_dev:
    :return:
    """
    data_len = len(x_dev)
    batch_eval = dataHelper.batch_iter_eval(x_dev, y_dev, batch_size=32, num_epochs=1, shuffle=False)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch_eval, y_batch_eval in batch_eval:
        batch_len = len(x_batch_eval)
        feed_dict = {
            model.input_x: x_batch_eval,
            model.input_y: y_batch_eval,
            model.dropout_keep_prob: 1.0
        }
        loss, accuracy = sess.run(
            [model.loss, model.accuracy],
            feed_dict)
        total_loss += loss * batch_len
        total_acc += accuracy * batch_len
    # time_str = datetime.datetime.now().isoformat()
    # print("{}: loss {:g}, acc {:g}".format(time_str, total_loss / data_len, total_acc / data_len))
    return total_loss / data_len, total_acc / data_len


def train():
    print("Configuring TensorBoard and Saver ...")
    tensorboard_dir = FLAGS.tensorboard_dir
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    # Configuring Saver
    saver = tf.train.Saver()
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    # Load data
    print("Loading data...")
    start_time = time.time()
    x_train, y_train, x_dev, y_dev, x_test, y_test = dataHelper.load_data(FLAGS.train_data_file, FLAGS.dev_sample_percentage, FLAGS.test_sample_percentage)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # Create Session
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and deviation ...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_dev = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 5000  # 如果超过5000论未提升，提前结束训练

    tag = False
    for epoch in range(FLAGS.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = dataHelper.batch_iter_per_epoch(x_train, y_train, FLAGS.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, FLAGS.dropout_keep_prob)
            if total_batch % FLAGS.checkpoint_every == 0:
                # write to tensorboard scalar
                summary = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(summary, total_batch)

            if total_batch % FLAGS.evaluate_every == 0:
                # print performance on train set and dev set
                feed_dict[model.dropout_keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.accuracy], feed_dict=feed_dict)
                loss_dev, acc_dev = evaluate(x_dev, y_dev, session)

                if acc_dev > best_acc_dev:
                    # save best result
                    best_acc_dev = acc_dev
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                print('Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%}, Val Loss: {3:>6.2}, ''Val Acc: '
                      '{4:>7.2%}, Time: {5} {6}'
                      .format(total_batch, loss_train, acc_train, loss_dev, acc_dev, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # having no improvement for a long time
                print("No optimization for a long time, auto-stopping...")
                tag = True
                break
        if tag:  # early stopping
            break


def test():
    print("Loading test data ...")
    x_train, y_train, x_dev, y_dev, x_test, y_test = dataHelper.load_data(FLAGS.train_data_file,
                                                                          FLAGS.dev_sample_percentage,
                                                                          FLAGS.test_sample_percentage)
    start_time = time.time()

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, save_path=save_path)

    print('Testing ...')
    loss_test, acc_test = evaluate(x_test, y_test, session)
    print('Test loss: {0:>6.2}, Test acc: {1:>7.2%}'.format(loss_test, acc_test))

    x_test_batches = dataHelper.batch_iter(x_test, batch_size=FLAGS.batch_size, num_epochs=1, shuffle=False)
    all_predictions = []
    # all_predict_prob = []
    count = 0  # concatenate第一次不能为空，需要加个判断来赋all_predict_prob值
    for x_test_batch in x_test_batches:
        batch_predictions = session.run(model.y_pred, feed_dict={
                                                                model.input_x: x_test_batch,
                                                                model.dropout_keep_prob: 1.0
                                                            })
        # all_predictions = np.concatenate([all_predictions, batch_predictions])

        if count == 0:
            all_predictions = batch_predictions
        else:
            all_predictions = np.concatenate([all_predictions, batch_predictions])
        count = 1
    print(all_predictions.shape)
    print(y_test.shape)
    """
    # One hot encoding
    # ohe = OneHotEncoder()
    # ohe.fit([[0], [1], [2], [3], [4]])
    # y_test_onehot = ohe.transform(y_test.reshape(-1, 1)).toarray()
    """
    all_predictions = np.argmax(all_predictions, axis=1)
    print(all_predictions.shape)
    y_test = y_test.reshape(-1)
    print(y_test.shape)
    # Evaluation indexes
    # y_test = np.argmax(y_test, axis=1)
    print("Precision, Recall, F1-Score ...")
    print(metrics.classification_report(y_test, all_predictions, target_names=['x', 's', 'b', 'm', 'e']))
    # Confusion Matrix
    print("Confusion Matrix ...")
    print(metrics.confusion_matrix(y_test, all_predictions))

    print("One example of word segmentation:")
    print("")

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("Please input: python3 runBiLSTM.py [train/test]")

    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    model = BiLSTM(
        timestep_size=FLAGS.timestep_size,
        tag_class=FLAGS.num_classes,
        vocab_size=FLAGS.vocab_size,
        embedding_dim=FLAGS.embedding_dim,
        num_layers=FLAGS.num_layers,
        hidden_dim=FLAGS.hidden_dim,
        learning_rate=FLAGS.learning_rate
    )
    if sys.argv[1] == 'train':
        train()
    else:
        test()
