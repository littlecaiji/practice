import numpy as np


def sigmoid(z):
    return 1.0/(1.0 + np.exp(z))


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
            a = sigmoid(w*a+b)
        return a

    def SGD(self, training_data, epochs, bathc_size, eta, test_data=None):
        if test_data != None:
            n_test = len(test_data)
        n = len(training_data)
        # 进行epochs次循环
        for j in range(epochs):
            np.random.shuffle(training_data)
            # 将training data切分
            mini_batches = [training_data[k:k+bathc_size]
                          for k in range(0, n, bathc_size)]
            for mini_batch in mini_batches:
                #self.update_mini_batch(mini_batch,eta)
                pass
            if test_data != None:
                #print("Epoch {} : {}/{}".format(j,self.evaluate(test_data),n_test))
                pass
            else:
                print("Epoch {} Complete".format(j))

    def update_mini_batch(self,mini_batch,eta):
        pass
