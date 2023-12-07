import os

import torch
from d2l import torch as d2l


# #@save
# d2l.DATA_HUB['aclImdb'] = (
# 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
# '01ada507287d82875905620988597833ad4e0903')
# data_dir = d2l.download_extract('aclImdb', 'aclImdb')
# data_dir = d2l.download_extract('aclImdb', 'aclImdb')
# @save
def read_imdb(data_dir, is_train):
    """读取IMDb评论数据集⽂本序列和标签"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test', label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels


def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    # fname = download(name)
    base_dir = os.path.dirname('../data/aclImdb')
    data_dir, ext = os.path.splitext('../data/aclImdb')
    return os.path.join(base_dir, folder) if folder else data_dir


# @save
def load_data_imdb(batch_size, num_steps=200):
    """返回数据迭代器和IMDb评论数据集的词表"""
    print("开始读取训练数据...")
    data_dir = download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    print("训练数据读取完毕...")
    # test_data = read_imdb(data_dir, False)
    print("测试数据读取完毕...")
    train_tokens = d2l.tokenize(train_data[0], token='word')
    # test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_iter = None
    # test_features = torch.tensor([d2l.truncate_pad(
    #     vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size, is_train=True)
    print("训练数据预处理完毕...")
    # test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
    #                            batch_size, is_train=False)
    return train_iter, test_iter, vocab