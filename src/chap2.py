# coding=utf-8

"""
    将chap1.py中的mini_batch方法改为矩阵而非for循环实现
"""

# coding=utf-8
import numpy as np
import random

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


class Network(object):

    def __init__(self, sizes):
        """
            sizes = [5,6,3,1]
            输入层有5个单元
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # zip函数将两个可迭代的对象中的元素一一对应起来，并返回一个统一的可迭代对象。
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, bathc_size, eta, test_data=None):
        if test_data != None:
            n_test = len(test_data)
        n = len(training_data)
        # 进行epochs次循环
        for j in range(epochs):
            random.shuffle(training_data)
            # 将training data切分
            mini_batches = [training_data[k:k+bathc_size]
                            for k in range(0, n, bathc_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data != None:
                print("Epoch {} : {}/{}".format(j,
                                                self.evaluate(test_data), n_test))
                pass
            else:
                print("Epoch {} Complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # mini_batch是一个list，长度为batch_size 10
        # mini_batch[0][0]是第一个batch的x向量，mini_batch[0][1]是第一个batch的y向量
        inputX_size = mini_batch[0][0].shape[0]
        inputY_size = mini_batch[0][1].shape[0]
        batch_size = len(mini_batch)
        x_matrix = np.zeros(shape=(inputX_size, batch_size))
        y_matrix = np.zeros(shape=(inputY_size, batch_size))
        count = 0
        for x, y in mini_batch:
            x_matrix[:, count] = x.flatten()
            y_matrix[:, count] = y.flatten()
            count += 1
        nabla_b, nabla_w = self.backprop(x_matrix, y_matrix)

        self.weights = [w-(eta)*nw for w,
                        nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta)*nb for b,
                       nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        '''
            (data_size,batch_size) x.shape = (784,10)  y.shape = (10,10)
        '''
        batch_size = x.shape[1]
        # 生成保存梯度的numpy array
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向传播
        activation = x  # 每次循环的激活值
        activations = []  # 所有层的激活值
        zs = []  # 所有层的带权重输入值
        activations.append(activation)
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(
            activations[-1], y) * sigmoid_prime(zs[-1])  # 输出层误差
        delta_mean = delta.sum(axis=1, keepdims=True) / batch_size
        nabla_b[-1] = delta_mean
        w_tmp = np.dot(delta, activations[-2].transpose())
        nabla_w[-1] = w_tmp / batch_size
        for l in xrange(2, self.num_layers):
            delta = np.dot(self.weights[-l+1].transpose(),
                           delta) * sigmoid_prime(zs[-l])
            delta_mean = delta.sum(axis=1, keepdims=True) / batch_size
            nabla_b[-l] = delta_mean
            w_tmp = np.dot(delta, activations[-l-1].transpose())
            nabla_w[-l] = w_tmp / batch_size

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
