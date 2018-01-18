# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import cia_inference

#配置神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZITION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

#模型保存的路径和文件名
MODEL_SAVE_PATH = "/path/to/model/"
MODEL_NAME = "model.ckpt"

def train(minst):
    #定义输入输出placeholder
    x = tf.placeholder(tf.float32, [
                    BATCH_SIZE,
                    cia_inference.IMAGE_SIZE,
                    cia_inference.IMAGE_SIZE,
                    cia_inference.NUM_CHANNELS],
                    name = 'x-input')
    y_ = tf.placeholder(tf.float32, [BATCH_SIZE, cia_inference.OUTPUT_NODE])
    #将输入的训练数据格式调整为一个四维矩阵
    # reshaped_xs = np.reshape(xs, (
    #     BATCH_SIZE,
    #     cia_inference.IMAGE_SIZE,
    #     cia_inference.IMAGE_SIZE,
    #     cia_inference.NUM_CHANNELS))

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZITION_RATE)

    #使用cia.inference中的前向传播
    y = cia_inference.inference(x, 1, regularizer)
    global_step = tf.Variable(0, trainable = False)

    #定义损失函数，学习率，滑动平均操作以及训练过程

    #使用滑动平均（暂时不懂啥意思，不知道这是在干嘛）
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    #定义交叉熵为损失函数并计算交叉熵平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #加正则化作为总损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    #定义衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        50000 / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    #批量随机梯度下降算法优化loss总损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                    .minimize(loss, global_step = global_step)

    #通过反向传播更新神经网络参数和参数的滑动平均值
    #等价于train_op = tf.group(train_step, variables_averages_op)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name = 'train')

    #初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_tables().run()

        #训练
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 将输入的训练数据格式调整为一个四维矩阵
            xs = np.reshape(xs, (
                BATCH_SIZE,
                cia_inference.IMAGE_SIZE,
                cia_inference.IMAGE_SIZE,
                cia_inference.NUM_CHANNELS))

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict = {x : xs, y_ : ys})

            #每1000轮保存一次模型
            if i % 10 == 0:
                #输出当前训练BATCH的损失
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                #保存当前模型。注意这里给出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数
                #如“model.ckpt-1000”表示训练1000轮之后得到的模型
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step = global_step)

# def main(argv = None):
#     mnist = input_data.read_data_sets("/MNIST_DATA", one_hot = True)
#     print('KO')
#     train(mnist)
#
# if __name__ == '__main__':
#     tf.app.run()
mnist = input_data.read_data_sets("MNIST_DATA", one_hot = True)
train(mnist)