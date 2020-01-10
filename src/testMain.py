# coding=utf-8

import mnist_loader
import chap2 as network
import os

os.chdir("/home/p310/PycharmProjects/practice/src")
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 196 ,30,10])

net.SGD(training_data, 30, 25, 3.0, test_data=test_data)