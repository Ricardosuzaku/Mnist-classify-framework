# -*- coding: utf-8 -*-
# 神经网络前向传播过程

import tensorflow as tf

# 配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABLES = 10

# 第一层卷积层尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层尺寸和深度
CONV2_DEEP = 32
CONV2_SIZE = 5
# 全连接层节点个数
FC_SIZE = 512

# 定义卷积神经网络前向传播过程
# 添加参数train用于区分训练过程和测试过程，训练中用到dropout方法防止过拟合

# 第一层卷积层变量并实现前向传播
# 过滤器权重，偏差初始化
def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP], initializer = tf.constant_initializer(0.0))

    # 过滤器尺寸为5 * 5 * 1 * 32，移动步长为1，全0填充。输出28 * 28 * 32的矩阵
    conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides = [1, 1, 1, 1], padding = 'SAME')
    # 使用relu激活函数
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第一个池化层前向传播。使用最大池化，过滤器边长为2，移动步长为2
    # 本层输入为上层输出，为28 * 28 * 32的矩阵。输出为14 * 14 * 32的矩阵
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    #声明第二层卷积层变量并实现前向传播。这一层输入为14 * 14 * 32的矩阵，输出为14 * 14 * 64的矩阵
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer = tf.truncated_normal_initializer(stddev = 0.1))
        conv2_biases = tf.get_variable(
            "bias", [CONV2_DEEP], initializer = tf.constant_initializer(0.0))

    # 过滤器尺寸为5 * 5 * 1 * 64，移动步长为1，全0填充。输出14 * 14 * 64的矩阵
    conv2 = tf.nn.conv2d(pool1, conv2_weights, strides = [1, 1, 1, 1], padding = 'SAME')

    # 使用relu激活函数
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第二个池化层前向传播。使用最大池化，过滤器边长为2，移动步长为2
    # 本层输入为上层输出，为14 * 14 * 64的矩阵。输出为7 * 7 * 64的矩阵
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    #将上一层7 * 7 * 32的矩阵输出拉直成一列向量进入全连接层
    #pool2.get_shape函数可以得到第四层输出矩阵维度而不需要手工计算
    #总共为四维，pool_shape[0]是第一维，表示batch个数
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    #通过tf.reshape函数将其变为batch * node的二维向量
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    #声明第五层全连接层变量并实现前向传播。输入为拉直的一组向量，维度为3136
    #输出为512维的向量。在全连接训练过程引入dropout防止过拟合
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weights", [nodes, FC_SIZE],
                                      initializer = tf.truncated_normal_initializer(stddev = 0.1))
        #全连接中加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer = tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5) #使用dropout

    #声明第六层全连接层变量并实现前向传播过程。这一层输入为一组长度为512的向量
    #输出为一组长度为10的向量。这一层输出通过softmax之后得到分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weights", [FC_SIZE, NUM_LABLES],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 全连接中加入正则化
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABLES], initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    #返回第六层输出
    return logit

