# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 16:07:29 2018

@author: Administrator
"""
import numpy as np
import os
import json
import pickle
import gensim
import random
from gensim.models import word2vec
from os.path import  exists, split
from keras.utils import np_utils
#from keras.models import Sequential, Model
#from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
#from keras.layers.merge import Concatenate
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import util



model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static
#model_type = "CNN-static"  # CNN-rand|CNN-non-static|CNN-static

# Model Hyperparameters
#embedding_dim = 100
embedding_dim = 500

filter_sizes = (3, 8)
#filter_sizes = (6, 8)
#num_filters = 128
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 32
#num_epochs = 10
num_epochs = 1

# Prepossessing parameters
#sequence_length = 1500   #400
sequence_length = 400   #400

#max_words = 20000   #5000

# Word2Vec parameters (see train_word2vec)
min_word_count = 1
context = 10




#
# ---------------------- Parameters end -----------------------
#def init():
#	f = open('utils/law.txt', 'r', encoding='utf8')
#	law = {}             # law{0: '184', 1: '336', 2: '314', ....}
#	lawname = {}         # lawname{184:0,336:2,...}
#	line = f.readline()
#	while line:
#		lawname[len(law)] = line.strip()
#		law[line.strip()] = len(law)
#		line = f.readline()
#	# print(lawname)
#	f.close()
#
#	f = open('utils/accu.txt', 'r', encoding='utf8')
#	accu = {}
#	accuname = {}
#	line = f.readline()
#	while line:
#		accuname[len(accu)] = line.strip()  # {{0: '妨害公务', 1: '寻衅滋事', 2: '盗窃、侮辱尸体'
#		accu[line.strip()] = len(accu)      # {'寻衅滋事': 1, '绑架': 8, ',...}
#		line = f.readline()
#	# print(accu)
#	f.close()
#
#	return law, accu, lawname, accuname
#
#law, accu, lawname, accuname = init()

#def get_time(time):
#	# 将刑期用分类模型来做
#	v = int(time['imprisonment'])
#
#	if time['death_penalty']:
#		return 0
#	if time['life_imprisonment']:
#		return 1
#	elif v > 10 * 12:
#		return 2
#	elif v > 7 * 12:
#		return 3
#	elif v > 5 * 12:
#		return 4
#	elif v > 3 * 12:
#		return 5
#	elif v > 2 * 12:
#		return 6
#	elif v > 1 * 12:
#		return 7
#	else:
#		return 8
#
#
#def get_label(d, kind):
#	global law
#	global accu
#
#	# 做单标签
#
#	if kind == 'law':
#		# 返回多个类的第一个
#		return law[str(d['meta']['relevant_articles'][0])]
#	if kind == 'accu':
#		return accu[d['meta']['accusation'][0]]
#
#	if kind == 'time':
#		return get_time(d['meta']['term_of_imprisonment'])

def slice_data(slice_size=None):
    if slice_size is None:
        alltext, accu_label, law_label, time_label = load_data()
#		alltext, accu_label, law_label, time_label = read_data()

    else:
#		alltext, accu_label, law_label, time_label = read_data()
        alltext, accu_label, law_label, time_label = load_data()
        randnum = random.randint(0,len(alltext))
        random.seed(randnum)
        random.shuffle(alltext)
        random.seed(randnum)
        random.shuffle(law_label)
        random.seed(randnum)
        random.shuffle(accu_label)
        random.seed(randnum)
        random.shuffle(time_label)
        alltext = alltext[:slice_size]
        law_label = law_label[:slice_size]
        accu_label = accu_label[:slice_size]
        time_label = time_label[:slice_size]
    return alltext, accu_label, law_label,time_label
#    return alltext, law_label, accu_label, time_label


#def read_data():
#	print('reading train data...')
#
#	train_data = []
##    with open('./cuttext_all_large.txt') as f:
#	with open('test_data/data_valid.json','r',encoding="utf-8") as f:
#		train_data = f.read().splitlines()
#	print(len(train_data))        # 154592
#
#	path = 'test_data/data_valid.json'
#	fin = open(path, 'r', encoding='utf-8')
#
#	accu_label = []
#	law_label = []
#	time_label = []
#
#	line = fin.readline()
#	while line:
#		d = json.loads(line)
#		accu_label.append(get_label(d, 'accu'))
#		law_label.append(get_label(d, 'law'))
#		time_label.append(get_label(d, 'time'))
#		line = fin.readline()
#	fin.close()
#
#	print('reading train data over.')
#	return train_data,accu_label, law_label, time_label


def load_data():
    train_fname='test_data/data_valid.json'
    """ load data from local file """
    facts = []
    accu_label = []
    article_label = []
    imprison_label = []
    k=0
    print('load data ing' )
    with open(train_fname,'r', encoding='utf-8') as f:
        line = f.readline()
#        while line and k<10:
        while line:
            k+=1
            line_dict = json.loads(line, encoding="utf-8")

            fact = line_dict["fact"]

            accu = util.get_label(line_dict, "accu")
            article = util.get_label(line_dict, "law")
            imprison = util.get_label(line_dict, "time")

            facts.append(fact)

            accu_label.append(accu)
            article_label.append(article)
            imprison_label.append(imprison)
            print('第'+str(k)+'个文档处理完！')
            line = f.readline()
    
    if util.DEBUG:
        print("DEBUG: training file loaded.")

#    facts = pd.Series(facts)
#    facts = facts.apply(util.cut_line)
    facts = [util.cut_line(line) for line in facts]


    if util.DEBUG:
        print("DEBUG: training data segmented.")

#    accu_label = pd.Series(accu_label)
#    article_label = pd.Series(article_label)
#    imprison_label = pd.Series(imprison_label)

    if util.DUMP:
        dump_processed_data_to_file(facts, accu_label, article_label, imprison_label)
    
    print('load_data sucess!')
    return facts, accu_label, article_label, imprison_label
#facts, accu_label, article_label, imprison_label=load_data()
#tok=text.Tokenizer()
#tok.fit_on_texts(facts)
#tok.word_index
#tok.texts_to_sequences(facts)

def dump_processed_data_to_file(self, facts, accu_label, article_label, imprison_label):
        """ dump processed data to `.pkl` file """
        data = [facts, accu_label, article_label, imprison_label]
        with open(util.MID_DATA_PKL_FILE_LOC, "wb") as f:
            pickle.dump(data, f)
        if util.DEBUG:
            print("DEBUG: data dumped to `.pkl` file")
            
            
def train_word2vec(sentence_matrix, vocabulary_inv,
                   num_features=300, min_word_count=1, context=10):
    """
    Trains, saves, loads Word2Vec model
    Returns initial weights for embedding layer.
    inputs:
    sentence_matrix # int matrix: num_sentences x max_sentence_len
    vocabulary_inv  # dict {int: str}
    num_features    # Word vector dimensionality
    min_word_count  # Minimum word count
    context         # Context window size
    """

    model_name = 'predictor/model/word2vec'
    if exists(model_name):
        # embedding_model = word2vec.Word2Vec.load(model_name)
        embedding_model = gensim.models.Word2Vec.load('predictor/model/word2vec')
        print('Load existing Word2Vec model \'%s\'' % split(model_name)[-1])
    else:
        # Set values for various parameters
        num_workers = 2  # Number of threads to run in parallel
        downsampling = 1e-3  # Downsample setting for frequent words

        # Initialize and train the model
        print('Training Word2Vec model...')
        sentences = [[vocabulary_inv[w] for w in s] for s in sentence_matrix]
        embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                            size=num_features, min_count=min_word_count,
                                            window=context, sample=downsampling)

        # If we don't plan to train the model any further, calling
        # init_sims will make the model much more memory-efficient.
        embedding_model.init_sims(replace=True)

        # Saving the model for later use. You can load it later using Word2Vec.load()
        print('Saving Word2Vec model \'%s\'' % split(model_name)[-1])
        embedding_model.save(model_name)

    # add unknown words
    embedding_weights = {key: embedding_model[word] if word in embedding_model else
    np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
                         for key, word in embedding_model.wv.vocab.items()}
    return embedding_weights

def data_process():
    train_data, accu_label, law_label, time_label = slice_data(1000000)
    # 转换成词袋序列
    maxlen = 500
    # 词袋模型的最大特征束
    max_features = 20000
    #生成Word2Vec模型
    model = word2vec.Word2Vec(train_data,size=maxlen,min_count=1)
    print('生成Word2Vec模型success!')
    model.save('predictor/model/word2vec')
    # 设置分词最大个数 即词袋的单词个数
#    with open('predictor/model/tokenizer.pickle', 'rb') as f:
#        tokenizer = pickle.load(f)
    tokenizer = Tokenizer(num_words=max_features, lower=True)  # 建立一个max_features个词的字典
    tokenizer.fit_on_texts(train_data)  # 使用一系列文档来生成token词典，参数为list类，每个元素为一个文档。可以将输入的文本中的每个词编号，编号是根据词频的，词频越大，编号越小。
    global word_index
    word_index = tokenizer.word_index  # 长度为508242
    sequences = tokenizer.texts_to_sequences(
        train_data)  # 对每个词编码之后，每个文本中的每个词就可以用对应的编码表示，即每条文本已经转变成一个向量了 将多个文档转换为word下标的向量形式,shape为[len(texts)，len(text)] -- (文档数，每条文档的长度)
    x = sequence.pad_sequences(sequences, maxlen,dtype='int16')  # 将每条文本的长度设置一个固定值。
    del tokenizer, sequences
    y = np_utils.to_categorical(accu_label)  # 多分类时，此方法将1，2，3，4，....这样的分类转化成one-hot 向量的形式，最终使用softmax做为输出
    print(x.shape, y.shape)
    indices = np.arange(len(x))
    lenofdata = len(x)
    np.random.shuffle(indices)
    #训练集和测试集比例8:2
    x_train = x[indices][:int(lenofdata * 0.8)]
    y_train = y[indices][:int(lenofdata * 0.8)]
    x_test = x[indices][int(lenofdata * 0.8):]
    y_test = y[indices][int(lenofdata * 0.8):]


    
    model = word2vec.Word2Vec.load('predictor/model/word2vec')
#    word2idx = {"_PAD": 0}  # 初始化 `[word : token]` 字典，后期 tokenize 语料库就是用该词典。
#    vocabulary_inv = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    vocabulary_inv = dict((k, model.wv[k]) for k, v in model.wv.vocab.items())

    return x,y,x_train, y_train, x_test, y_test, vocabulary_inv

#
#==============================================================================
# ## Data Preparation
#==============================================================================
print("Load data...")
x,y,x_train, y_train, x_test, y_test, vocabulary_inv = data_process()
#
w = train_word2vec(x, vocabulary_inv)

if sequence_length != x_test.shape[1]:
    print("Adjusting sequence length for actual size")
    sequence_length = x_test.shape[1]

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("Vocabulary Size: {:d}".format(len(vocabulary_inv)))

# Prepare embedding layer weights and convert inputs for static model
print("Model type is", model_type)
if model_type in ["CNN-non-static", "CNN-static"]:
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                       min_word_count=min_word_count, context=context)
    if model_type == "CNN-static":
        x_train = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_train])
        x_test = np.stack([np.stack([embedding_weights[word] for word in sentence]) for sentence in x_test])
        print("x_train static shape:", x_train.shape)
        print("x_test static shape:", x_test.shape)

elif model_type == "CNN-rand":
    embedding_weights = None
else:
    raise ValueError("Unknown model type")


    
    
    
    
    
with open('embedding_weights.pkl','wb')as f:
    pickle.dump(embedding_weights,f)
    print('embedding_weights'+'存入文件成功')

with open('x_test.pkl','wb')as f:
    pickle.dump(x_test,f) 
    print('x_test'+'存入文件成功')
    
with open('y_test.pkl','wb')as f:
    pickle.dump(y_test,f)
    print('y_test'+'存入文件成功')

with open('y_train.pkl','wb')as f:
    pickle.dump(y_train,f)  
    print('y_train'+'存入文件成功')
    
with open('x_train.pkl','wb')as f:
    pickle.dump(x_train,f)
    print('x_train'+'存入文件成功')

     
  