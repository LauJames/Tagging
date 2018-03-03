#! /user/bin/evn python
# -*- coding:utf8 -*-

"""

@Author   : Lau James
@Contact  : LauJames2017@whu.edu.cn
@Project  : Tagging 
@File     : dataHelper.py
@Time     : 2018/3/1 13:55
@Software : PyCharm
@Copyright: "Copyright (c) 2018 Lau James. All Rights Reserved"
"""

import numpy as np
import pandas as pd
import re
import codecs
import tensorflow as tf
import os
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter
from itertools import chain


def clean_str(string):
    if u'“/s' not in string:
        return string.replace(u'”/s', '')
    elif u'”/s' not in string:
        return string.replace(u'“/s', '')
    elif u'‘/s' not in string:
        return string.replace(u'’/s', '')
    else:
        return string


def join_data(file_path):
    """
    Reorganized corpus. Original data was divided by '/r/n'
    :param file_path:
    :return: sentences divided by '，。！？、‘’“”' , not by '\r\n'
    """
    # read all data in str type
    with codecs.open(file_path, encoding='gbk') as fp:
        texts = fp.readlines()
    # sentences = texts.split('\r\n') # cut the texts by \r\n

    texts = u''.join(map(clean_str, texts))  # combine all words
    print('Length of texts is %d' % len(texts))
    print('Example of texts: \n', texts[:100])

    # reorganize
    sentences = re.split(u'[，。！？、‘’“”]/[bems]', texts)
    print('Sentences number: %d' % len(sentences))
    print('Example of sentences:\n', sentences[0])
    return sentences


def get_words_tags(sentences):
    """
    process sentences into [word1, word2, ..., wordn], [tag1, tag2, ...,tagn]
    :param sentences:
    :return: [word1, word2, ..., wordn], [tag1, tag2, ...,tagn]
    """
    data_list = list()
    tags_list = list()
    for sentence in sentences:
        words_tags = re.findall('(.)/(.)', sentence)
        if words_tags:
            words_tags = np.array(words_tags)
            data_list.append(words_tags[:, 0])
            tags_list.append(words_tags[:, 1])
    print('Length of sentences is %d' % len(data_list))
    print('Example of data:', data_list[0])
    print('Example of tag:', tags_list[0])
    return data_list, tags_list


def char2id(data, tags_list, max_len=32):
    """
    Project words and tags into id (padding the sentences to a same length)
    Saving to .pkl file
    :param data:
    :param tags_list:
    :return:
    """
    df_data = pd.DataFrame({'words': data, 'tags': tags_list}, index=range(len(tags_list)))
    df_data['sentence_len'] = df_data['tags'].apply(lambda tags: len(tags))
    all_words = list(chain(*df_data['words'].values))

    # count all the words
    allwords_series = pd.Series(all_words)
    allwords_series = allwords_series.value_counts()
    set_words = allwords_series.index
    set_ids = range(1, len(set_words) + 1)  # start from 1, 0 using as padding
    tags = ['x', 's', 'b', 'm', 'e']
    tag_ids = range(len(tags))

    word2id = pd.Series(set_ids, index=set_words)
    id2word = pd.Series(set_words, index=set_ids)
    tag2id = pd.Series(tag_ids, index=tags)
    id2tag = pd.Series(tags, index=tag_ids)

    vocab_size = len(set_words)
    print('vocab_size = %d' % vocab_size)

    def x_padding(words):
        """
        :param words:
        :return: ids
        """
        ids = list(word2id[words])
        if len(ids) >= max_len:  # if it is longer , cut it off
            return ids[:max_len]
        ids.extend([0] * (max_len-len(ids)))  # if it is short, filled it
        return ids

    def y_padding(tags):
        """
        :param tags:
        :return:
        """
        ids = list(tag2id[tags])
        if len(ids) >= max_len:
            return ids[:max_len]
        ids.extend([0] * (max_len-len(ids)))
        return ids

    df_data['x'] = df_data['words'].apply(x_padding)
    df_data['y'] = df_data['tags'].apply(y_padding)

    x = np.asarray(list(df_data['x'].values))
    y = np.asarray(list(df_data['y'].values))
    print('X.shape={}, y.shape={}'.format(x.shape, y.shape))
    print('Example of words: ', df_data['words'].values[0])
    print('Example of X: ', x[0])
    print('Example of tags: ', df_data['tags'].values[0])
    print('Example of y: ', y[0])
    save_data(x, y, word2id, id2word, tag2id, id2tag)


def save_data(x, y, word2id, id2word, tag2id, id2tag):
    with open('data.pkl', 'wb') as output:
        pickle.dump(x, output)
        pickle.dump(y, output)
        pickle.dump(word2id, output)
        pickle.dump(id2word, output)
        pickle.dump(tag2id, output)
        pickle.dump(id2tag, output)
    print(' Finished saving the data.')


def load_data(data_path, test_pct, dev_pct):
    with open(data_path, 'rb') as inp:
        x = pickle.load(inp)
        y = pickle.load(inp)
        word2id = pickle.load(inp)
        id2word = pickle.load(inp)
        tag2id = pickle.load(inp)
        id2tag = pickle.load(inp)
    print('Loading successful!')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_pct, random_state=42)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=dev_pct, random_state=42)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def batch_iter_eval(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    :param x:
    :param y:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """
    x = np.asarray(x)
    y = np.asarray(y)
    data_size = len(x)
    num_batch_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_x = x[shuffle_indices]
            shuffle_y = y[shuffle_indices]
        else:
            shuffle_x = x
            shuffle_y = y
        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_x[start_index: end_index], shuffle_y[start_index: end_index]


def batch_iter_per_epoch(x, y, batch_size=64):
    """生成批次数据,每个epoch"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def batch_iter(data, num_epochs, batch_size=64, shuffle=True):
    """
        Generates a batch iterator for a dataset.
        :param data:
        :param batch_size:
        :param num_epochs:
        :param shuffle:
        :return:
        """
    data = np.array(data)
    data_size = len(data)
    num_batch_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = data[shuffle_indices]
        else:
            shuffle_data = data
        for batch_num in range(num_batch_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffle_data[start_index: end_index]


if __name__ == '__main__':
    # data = join_data('msr_train.txt')
    # data_list, tags_list = get_words_tags(data)
    # char2id(data_list, tags_list, max_len=32)
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data('data.pkl', 0.2, 0.2)
    print('X_train.shape={}, y_train.shape={}; \nX_valid.shape={}, y_valid.shape={};\nX_test.shape={}, y_test.shape={}'.format(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape))

    """
    # describable statistic
    length_list = list()
    for tags in tags_list:
        length_list.append(len(tags))
    c = Counter()
    for i in length_list:
        c[i] = c[i] + 1
    print('length frequncy stat:', str(c))
    len_pd = pd.Series(length_list)
    meanList = len_pd.mean()
    maxList = len_pd.max()
    minList = len_pd.min()
    medianList = len_pd.median()
    countList = len_pd.count()
    quantileList = len_pd.quantile([0.25, 0.75])
    print('长度均值:' + str(meanList))
    print('长度最大值:' + str(maxList))
    print('长度最小值:' + str(minList))
    print('长度中位数:' + str(medianList))
    print('1/4分位数、3/4分位数：\n' + str(quantileList))
    """
