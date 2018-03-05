#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Tagging 
@File     : utest.py
@Time     : 2018/3/2 16:25
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import re
import time
import datetime
import pickle
import numpy as np
import tensorflow as tf
from model.BiLSTM import BiLSTM


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def x_padding(ids, max_len=32):
    """
    :param words:
    :return: ids
    """
    if len(ids) >= max_len:  # if it is longer , cut it off
        return ids[:max_len]
    ids.extend([0] * (max_len-len(ids)))  # if it is short, filled it
    return ids


def word2id(word):
    with open('./data/data.pkl', 'rb') as inp:
        x = pickle.load(inp)
        y = pickle.load(inp)
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
    return word2id[word]

def id2tag(id):
    with open('./data/data.pkl', 'rb') as inp:
        x = pickle.load(inp)
        y = pickle.load(inp)
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
    return id2tag[id]


def reorganized_data(sentence):
    sentences = re.split(u'[，。！？、‘’“”]', sentence)
    words = list()
    for sentence in sentences:
        print(sentence + '\n')
    return sentences


if __name__ == "__main__":
    """
    tags = ['s', 'b',  'e', 's', 's', 'b', 'm', 'e']
    words = ['他', '今', '天', '吃', '了', '马', '铃', '薯']
    rss = ''
    for i in range(len(tags)):
        if tags[i] in ['s', 'e']:
            rss = rss + words[i] + ' '
        else:
            rss = rss + words[i]
    print(rss)
    """

    start_time = time.time()

    model = BiLSTM(
        timestep_size=32,
        tag_class=5,
        vocab_size=5159,
        embedding_dim=256,
        num_layers=2,
        hidden_dim=128,
        learning_rate=0.001
    )

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, save_path='checkpoints/BiLSTM/best_validation')

    test = '人们常说，生活是一部教科书。'
    x_test_raw = reorganized_data(test)
    textID = list()
    sentenceID = list()
    for text in x_test_raw:
        if text:
            text_len = len(text)
            words = list()
            for word in text:
                textID.append(word2id(word))
                words.append(word)
            textID = np.asarray(textID)
            predict = session.run(model.y_pred, feed_dict={
                model.input_x: sentenceID,
                model.dropout_keep_prob: 1.0
            })
            textID = list()
            predict_tags = np.argmax(predict, axis=1)[:text_len]  # padding部分直接丢弃
            tags = list()
            for id in predict_tags:
                tags.append(id2tag(id))
            rss = ''
            for i in range(len(tags)):
                if tags[i] in ['s', 'e']:
                    rss = rss + words[i] + ' '
                else:
                    rss = rss + words[i]
            print(rss)
