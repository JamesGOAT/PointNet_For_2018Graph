# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 20:44:16 2018

@author: 陈家栋
"""
import tensorflow as tf

def load_model():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('log/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint("log/"))
        
b = 1

print(a+b)