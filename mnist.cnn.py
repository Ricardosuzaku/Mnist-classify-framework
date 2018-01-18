import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

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
BATCH_SIZE = 100

mnist = input_data.read_data_sets("MNIST_DATA", one_hot = True)
sess = tf.InteractiveSession()

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, [None, INPUT_NODE])
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

W_fc1 = weight_variable([7 * 7 * 64, FC_SIZE])
b_fc1 = bias_variable([FC_SIZE])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([FC_SIZE, NUM_LABLES])
b_fc2 = bias_variable([NUM_LABLES])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #tf.cast将数据转换成指定类型

# 开始训练
# tf.global_variables_initializer().run()
sess.run(tf.initialize_all_variables())

for i in range(20000):
    batch = mnist.train.next_batch(BATCH_SIZE)
    # print(batch[0].shape)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    if i % 100 == 0:
        # print("w", W_conv1.eval())
        train_accuracy = accuracy.eval(feed_dict = {x: mnist.test.images[0: 200],
                                                    y_: mnist.test.labels[0: 200], keep_prob: 1.0})
        loss_value = cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print("step %d, trainning accuracy %g， loss %g" % (i, train_accuracy, loss_value))


# 测试
# print("test accuracy %g" % accuracy.eval(feed_dict = {x:mnist.test.images[0: 5000],
#                                                       y_:mnist.test.labels[0: 5000], keep_prob: 1.0}))
t_accuracy = []
for i in range(10):
    batch_test = mnist.test.next_batch(1000)
    t_accuracy += [accuracy.eval(feed_dict={x: batch_test[0], y_: batch_test[1], keep_prob: 1.0})]
    # print("step %d test accuracy %g" % (i, accuracy.eval(feed_dict = {x:batch_test[0],
    #                                                                   y_:batch_test[1], keep_prob: 1.0})))
print("test accuracy %g" % np.array(t_accuracy).mean())




