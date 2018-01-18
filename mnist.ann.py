import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_DATA", one_hot = True)
sess = tf.InteractiveSession()

# 定义神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10
HIDEN_NODE = 300
BATCH_SIZE = 100
lamda = 0.004

# 定义前向传播
x = tf.placeholder(tf.float32, [None, INPUT_NODE])
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])
keep_prob = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.truncated_normal([INPUT_NODE, HIDEN_NODE], stddev = 0.1))
b1 = tf.Variable(tf.constant(0.1, shape = [HIDEN_NODE]))
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)
layer1_drop = tf.nn.dropout(layer1, keep_prob)

w2 = tf.Variable(tf.truncated_normal([HIDEN_NODE, OUTPUT_NODE], stddev = 0.1))
b2 = tf.Variable(tf.constant(0.1, shape = [OUTPUT_NODE]))
y = tf.nn.softmax(tf.matmul(layer1_drop, w2) + b2)

w1_loss = lamda * tf.nn.l2_loss(w1)
w2_loss = lamda * tf.nn.l2_loss(w2)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))
loss = w1_loss + w2_loss + cross_entropy
# 优化器有GradientDescentOptimizer、AdagradOptimizer、AdagradDAOptimizer、MomentumOptimizer
# AdamOptimizer、FtrlOptimizer、RMSPropOptimizer等
train_step = tf.train.AdagradOptimizer(0.03).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #tf.cast将数据转换成指定类型

# 开始训练
# tf.global_variables_initializer().run()
sess.run(tf.initialize_all_variables())

for i in range(20000):
    x_train = mnist.train.next_batch(BATCH_SIZE)
    train_step.run(feed_dict = {x: x_train[0], y_: x_train[1], keep_prob: 0.75})
    if i % 100 == 0:
        # train_accuracy = accuracy.eval(feed_dict = {x: x_train[0], y_: x_train[1], keep_prob: 0.75})
        # train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images[0: 200],
        #                                           y_: mnist.test.labels[0: 200], keep_prob: 1.0})
        train_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
        loss_value = cross_entropy.eval(feed_dict={x: x_train[0], y_: x_train[1], keep_prob: 0.75})
        y_value = y.eval(feed_dict={x: x_train[0], y_: x_train[1], keep_prob: 0.75})
        # loss_value = loss.eval(feed_dict={x: x_train[0], y_: x_train[1], keep_prob: 0.75})
        # print("step %d, y %g" % (i, y_value))
        print("step %d, trainning accuracy %g, loss %g" % (i, train_accuracy, loss_value ))

# 测试
# x_test = mnist.test.next_batch(BATCH_SIZE)
test_accuracy = accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
print("test accuracy %g" % test_accuracy)