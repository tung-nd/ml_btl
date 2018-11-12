#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 11:22:41 2018

@author: tungnd
preprocess data and compute tf-idf
"""
from os import listdir
from os.path import join, isfile
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from collections import defaultdict

def gather_20newsgroups_data():
    path = './20news-bydate'
    # dirs là đường dẫn 2 thư mục train và test
    dirs = [join(path, dir_name) for dir_name in listdir(path) if not isfile(join(path, dir_name))]
    train_dir, test_dir = (dirs[0], dirs[1]) if 'train' in dirs[0] else (dirs[1], dirs[0])
    # lấy ra tên các topic, chính là tên các thư mục trong thư mục train, có 20 topic
    topic_list = [topic for topic in listdir(train_dir)]
    topic_list.sort()
    
    stop_words = stopwords.words("english") # stop_words là danh sách các stop words
    ps = PorterStemmer()    # dùng để steam word => đưa word về dạng origin
    
    def collect_data(parent_dir, topic_list):
        data = []
        # với từng topic trong list
        for topic_id, topic in enumerate(topic_list):
            label = topic_id # nhãn của topic
            topic_path = parent_dir + '/' + topic + '/' #đường dẫn của topic
            files = [(file_name, topic_path + file_name) \
            for file_name in listdir(topic_path) if isfile(topic_path + file_name)]
            # files là 1 list các tuple, mỗi tuple: (tên file, đường dẫn file)
            files.sort()
            # xử lý từng file
            for file_name, file_path in files:
                with open(file_path, errors = "ignore") as f:
                    text = f.read().lower()
                    words = word_tokenize(text, "english")  # tách từ
                    words = [w for w in words if w not in stop_words]   # loại bỏ stop_words
                    words = [ps.stem(w) for w in words if w.isalpha()]  # loại bỏ kí tự k phải chữ
                    content = ' '.join(words)   #nối lại thành 1 string
                    data.append(str(label) + '<ffff>' + file_name + '<ffff>' + content)
                    #data là 1 list các string, mỗi string có định dạng "topic_label<ffff>doc_id<ffff>doc_content"
        return data
        
    train_data = collect_data(train_dir, topic_list)
    test_data = collect_data(test_dir, topic_list)
    full_data = train_data + test_data
    
    # ghi data ra file
    with open("train-processed.txt", "w") as f:
        f.write('\n'.join(train_data))
        
    with open("test-processed.txt", "w") as f:
        f.write('\n'.join(test_data))
        
    with open("full-processed.txt", "w") as f:
        f.write('\n'.join(full_data))
        
def generate_vocab(data_path, min_df = 0.001, max_df = 0.5):    # min_df và max_df để loại bỏ những từ quá hiếm hoặc quá thông dụng
    def compute_idf(df, corpus_size):
        assert df > 0
        return np.log10(corpus_size * 1. / df)
    
    with open(data_path) as f:
        lines = f.read().splitlines()   # mỗi dòng là 1 văn bản
    corpus_size = len(lines)
    doc_count = defaultdict(int)
    for line in lines:
        features = line.split('<ffff>')
        words = features[-1]    # lấy ra nội dung
        words = list(set(words.split()))    #phải dùng set thì 1 từ xuất hiện 2 lần trong 1 văn bản thì cũng chỉ tăng df lên 1 => sử dụng set để loại bỏ trùng lặp
        for w in words:
            doc_count[w] += 1
    
    idfs = [(w, compute_idf(df, corpus_size)) for w, df in zip(doc_count.keys(), doc_count.values()) if df / corpus_size >= min_df and df / corpus_size <= max_df]
    idfs.sort(key = lambda tup: tup[1], reverse = True)
    print ("Vocab size: %d" %len(idfs))
    with open("vocab.txt", "w") as f:
        f.write('\n'.join([w + '<ffff>' + str(idf) for w, idf in idfs]))
    # mỗi từ trong vocab có định dạng "từ<ffff>idf"
        
def get_tf_idf(data_path, vocab_path):
    with open(vocab_path) as f:
        idfs = [(line.split('<ffff>')[0], float(line.split('<ffff>')[1])) for line in f.read().splitlines()]
        ids = dict([(word, index) for index, (word, df) in enumerate(idfs)])
        idfs = dict(idfs)
        
    with open(data_path) as f:
        documents = [(int(line.split('<ffff>')[0]), int(line.split('<ffff>')[1]), line.split('<ffff>')[2]) for line in f.read().splitlines()]
        
    idf_data = []
    for doc in documents:
        topic, doc_id, text = doc
        words = [w for w in text.split() if w in idfs]
        word_set = list(set(words))
        max_term_freq = max([words.count(word) for word in word_set])
    
        words_tfidfs = []
        sum_squares = 0.0
        for word in word_set:
            term_freq = words.count(word)
            tfidf_value = term_freq * 1. / max_term_freq * idfs[word]
            words_tfidfs.append((ids[word], tfidf_value))
            sum_squares += tfidf_value ** 2
            
        words_tfidf_normalized = [str(index) + ':' + str(idf / np.sqrt(sum_squares)) for index, idf in words_tfidfs]
        sparse_repr = ' '.join(words_tfidf_normalized)
        idf_data.append((topic, doc_id, sparse_repr))
        
    with open("tf-idf.txt", "w") as f:
        f.write('\n'.join([str(topic) + '<ffff>' + str(doc_id) + '<ffff>' + sparse_repr for topic, doc_id, sparse_repr in idf_data]))
            
gather_20newsgroups_data()
generate_vocab("train-processed.txt")
#get_tf_idf('./test-processed.txt', 'vocab.txt')

        
        


