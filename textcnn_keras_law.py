# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 10:35:55 2018

@author: Administrator
"""

import numpy as np
import os
import pickle
from keras.models import  Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding
from keras.layers.merge import Concatenate
np.random.seed(0)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
#import keras.backend.tensorflow_backend as KTF
 
#KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
 

gpu_options = tf.GPUOptions(allow_growth=True)

sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
 

#
##日志
#import sys
#f_handler=open('out.log', 'w')
#sys.stdout=f_handler

## ---------------------- Parameters section -------------------
##
## Model type. See Kim Yoon's Convolutional Neural Networks for Sentence Classification, Section 3
#model_type = "CNN-non-static"  # CNN-rand|CNN-non-static|CNN-static
##model_type = "CNN-static"  # CNN-rand|CNN-non-static|CNN-static
#
## Model Hyperparameters
##embedding_dim = 100
#embedding_dim = 500
#
#filter_sizes = (3, 8)
##filter_sizes = (6, 8)
##num_filters = 128
##num_filters = 10
#num_filters = 50
#dropout_prob = (0.5, 0.8)
#hidden_dims = 50
#
## Training parameters
##batch_size = 32
#batch_size = 64
#num_epochs = 50
##num_epochs = 1
#
## Prepossessing parameters
##sequence_length = 1500   #400
#sequence_length = 500
##sequence_length = 400   #400
#
##max_words = 20000   #5000
#
## Word2Vec parameters (see train_word2vec)
#min_word_count = 1
#context = 10
#
#  
#    
##以下是 模型需要的数据一共五个大文件，加上两个参数   
#with open('/input/x_test.pkl','rb')as f:
#    x_test=pickle.load(f)
#    print('x_test'+'载入文件成功')
#
#with open('/input/y_test.pkl','rb')as f:
#    y_test=pickle.load(f)
#    print('y_test'+'载入文件成功')
#
#    
#with open('/input/x_train.pkl','rb')as f:
#    x_train=pickle.load(f)
#    print('x_train'+'载入文件成功')
#
#    
#with open('/input/y_train.pkl','rb')as f:
#    y_train=pickle.load(f) 
#    print('y_train'+'载入文件成功')
#
#    
#with open('/input/embedding_weights.pkl','rb')as f:
#    embedding_weights=pickle.load(f) 
#    print('embedding_weights'+'载入文件成功')
#
# 
#len_vocabulary_inv=103943
#y_shape_1=202
#print("Load data success...")
##==============================================================================
## # Build model
##==============================================================================
#def text_cnn():
#    if model_type == "CNN-static":
#        input_shape = (sequence_length, embedding_dim)
#    else:
#        input_shape = (sequence_length,)
#    
#    model_input = Input(shape=input_shape)
#    
#    # Static model does not have embedding layer
##    嵌入层Embedding被定义为网络的第一个隐藏层。它必须指定3个参数：
##    它必须指定3个参数：
##    1.input_dim：这是文本数据中词汇的大小。 
##    2.output_dim：这是嵌入单词的向量空间的大小。它为每个单词定义了该层的输出向量的大小。例如，它可以是32或100甚至更大。根据你的问题来定。
##    3.input_length：这是输入序列的长度，正如你为Keras模型的任何输入层定义的那样。例如，如果你的所有输入文档包含1000个单词，则为1000
#    if model_type == "CNN-static":
#        z = model_input
#    else:
#        z = Embedding(len_vocabulary_inv, embedding_dim, input_length=sequence_length, name="embedding")(model_input)
#    
#    z = Dropout(dropout_prob[0])(z)
#    
#    # Convolutional block
#    conv_blocks = []
#    for sz in filter_sizes:
#        conv = Convolution1D(filters=num_filters,
#                             kernel_size=sz,
#                             padding="valid",
#                             activation="relu",
#                             strides=1)(z)
#        conv = MaxPooling1D(pool_size=2)(conv)
#        conv = Flatten()(conv)
#        conv_blocks.append(conv)
#    z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
#    
#    z = Dropout(dropout_prob[1])(z)
#    z = Dense(hidden_dims, activation="relu")(z)
#    model_output = Dense(y_shape_1, activation="softmax")(z)
#    
#    model = Model(model_input, model_output)
#    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#    #==============================================================================
#    # # Initialize weights with word2vec
#    #==============================================================================
#    if model_type == "CNN-non-static":
#        weights = np.array([v for v in embedding_weights.values()])
#        print("Initializing embedding layer with word2vec weights, shape", weights.shape)
#        embedding_layer = model.get_layer("embedding")
#        embedding_layer.set_weights([weights])
#
#    return model

##==============================================================================
## # Train the model
##==============================================================================
#print("training...")
#model=text_cnn()
#model.fit(x_train, y_train,validation_split=0.2, batch_size=batch_size, epochs=num_epochs, verbose=2,class_weight='auto')
#
#print("pridicting...")
#scores = model.evaluate(x_test,y_test)
#print('test_loss:%f,accuracy: %f'%(scores[0],scores[1]))
#
#print("saving accu_textcnnmodel")
#model.save('/data/textcnn_keras_law/accu_textcnn.h5')

from keras.models import load_model
#with open('accu_textcnn.h5')as f:
#accu_model=load_model('accu_textcnn.h5')
accu_model=load_model('accu_textcnn (3).h5')

#以下是 模型需要的数据一共五个大文件，加上两个参数   
with open('x_test.pkl','rb')as f:
    x_test=pickle.load(f)
    print('x_test'+'载入文件成功')

with open('y_test.pkl','rb')as f:
    y_test=pickle.load(f)
    print('y_test'+'载入文件成功')
    
with open('x_test.pkl','rb')as f:
    x_test=pickle.load(f)
    print('x_test'+'载入文件成功')

with open('y_test.pkl','rb')as f:
    y_test=pickle.load(f)
    print('y_test'+'载入文件成功')
    
print("pridicting...")
scores = accu_model.evaluate(x_test,y_test)
print('test_loss:%f,accuracy: %f'%(scores[0],scores[1]))