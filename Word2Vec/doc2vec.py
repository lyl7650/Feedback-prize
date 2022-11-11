# 直接生成sentence vector


import gensim, os
import numpy as np
from gensim.models.doc2vec import Doc2Vec



TaggededDocument = gensim.models.doc2vec.TaggedDocument
 
##需要运行pos_neg.txt、unverified.txt文件
def get_corpus():
    with open(path+'/train_text.txt', 'r', encoding='utf-8') as doc:  ####分词的数据生成的txt
        docs = doc.read().splitlines()
    train_docs = []
    for i, text in enumerate(docs):
        word_list = text.split(',')       ###分词的结果用“,”隔开
        length = len(word_list)
        word_list[length - 1] = word_list[length - 1].strip()
        document = TaggededDocument(word_list, tags=[i])
        train_docs.append(document)
    return train_docs
 
 
def train(x_train):
    ###vector_size 输出数值型数据纬度，epochs 迭代次数，workers 并行的核心数，缺少相应模块包不能并行
    model_dm = Doc2Vec(x_train, min_count=1, window=3, vector_size=300, sample=1e-3, negative=5, workers=5,epochs=300)
    model_dm.train(x_train, total_examples=model_dm.corpus_count,epochs=300)
    model_dm.save(path + '/model/sentence_vector')  #############model 保存
    model_dm.dv.save_word2vec_format(path+'/model/sentence_vector.txt')
    return model_dm
 
 
if __name__ == '__main__':
    path = os.getcwd()
    train_text = 'train_text.txt'
    model_name = 'test'
    x_train = get_corpus()
    print(x_train)
    model_dm = train(x_train)
