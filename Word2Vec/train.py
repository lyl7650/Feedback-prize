# coding:utf8
from gensim.models.word2vec import Word2Vec
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import segmentation, os, csv
import pandas as pd 

def data_process(df, train_text):
    """ 
        对text分词并存为list
        label为df['grammar']
    """
    data = pd.read_csv(path + '/data/' + df)
    output = data['cleantext']#.values.tolist()
    output.to_csv(path + '/data/' + train_text, index=False, header=False, quoting=csv.QUOTE_NONE,escapechar=' ')


def train(train_text, path, save_model_name):
    sentence = LineSentence(open(path+'/data/'+train_text,'r',encoding='utf8'))

    #min_count是最低出现数，默认数值是5；
    #vector_size是gensim Word2Vec将词汇映射到的N维空间的维度数量（N）默认的size数是100；
    #epoch是模型训练时在整个训练语料库上的迭代次数，假如参与训练的文本量较少，就需要把这个参数调大一些。iter的默认值为5；
    #sg是模型训练所采用的的算法类型：1 代表 skip-gram，0代表 CBOW，sg的默认值为0；
    #window控制窗口，如果设得较小，那么模型学习到的是词汇间的组合性关系（词性相异）；如果设置得较大，会学习到词汇之间的聚合性关系（词性相同）。模型默认的window数值为5；
    model = Word2Vec(sentences = sentence, vector_size=300, 
            alpha=0.002, window= 3, min_count=0, epochs = 5, sg=1)
    model.train(sentence, total_examples=model.corpus_count,epochs=300)
    model.save('./model/{}.model'.format(save_model_name))
    #model.save('./model/{}.bin'.format(save_model_name))
    model.wv.save_word2vec_format('./model/{}.txt'.format(save_model_name), binary=False)
    print("Model Saved Successfully!")


if __name__ == '__main__':
    path = os.getcwd()
    train_text = 'train_text.txt'
    model_name = 'test'
    data_process('train_clean.csv', train_text)
    train(train_text = train_text, path = path, save_model_name = model_name)

