import argparse
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000  # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        trainStats = {}

        ones_xTrain = np.ones((len(xTrain), 1))
        ones_xTest = np.ones((len(xTest), 1))
        xTrain = np.concatenate((ones_xTrain, xTrain), axis=1)
        xTest = np.concatenate((ones_xTest, xTest), axis=1)

        # Beta starts at 0 for simplicity
        self.beta = np.zeros((len(xTrain[0])))
        # Get number of batches
        num_batches = len(xTrain) // self.bs

        # Loop through Epochs
        for i in range(self.mEpoch):
            # Randomize Data
            #random_xTrain = np.random.permutation(xTrain)
            #random_yTrain = np.random.permutation(yTrain)
            random_xTrain, random_yTrain = shuffle(xTrain, yTrain)
            # Iterate through batches
            x_batch, y_batch = self.batch_processor(random_xTrain, random_yTrain)

            for j in range(len(x_batch)):
                start = time.time()
                x = x_batch[j]
                y = y_batch[j]

                # Get Gradient
                sum_gradient = np.matmul(x.transpose(), (y.transpose() - np.matmul(x, self.beta)).transpose())

                # Gradient update
                self.beta = self.beta + (self.lr / (len(x), 1)[0] * np.reshape(sum_gradient, len(self.beta)))

                train_mse = self.mse(xTrain, yTrain)
                test_mse = self.mse(xTest, yTest)

                end = time.time()
                timeElapsed = end - start
                trainStats[i * num_batches + j] = {}
                trainStats[i * num_batches + j]['time'] = timeElapsed
                trainStats[i * num_batches + j]['train-mse'] = train_mse
                trainStats[i * num_batches + j]['test-mse'] = test_mse

        return trainStats

    def batch_processor(self, x, y):
        length_batch = len(x) // self.bs
        x_batch = np.array_split(x, length_batch)
        y_batch = np.array_split(y, length_batch)
        return x_batch, y_batch

    def batch_size_comparison(self, xTrain, yTrain, xTest, yTest):
        self.lr = 0.01 # the initial optimal learning rate we obtained previously for bs =1 which is exactly the first bs in our list
        batch_size_list = [1]
        batch_size_list = batch_size_list + list(range(1677, len(xTrain)+1, 1677))


        list_train_mse_list = []
        list_test_mse_list = []
        list_time_list = []

        for batchsize in batch_size_list:

            train_mse_list = []
            time_list = []
            test_mse_list = []
            self.bs = batchsize
            trainStats = self.train_predict(xTrain, yTrain, xTest, yTest)
            for value in trainStats.values():
                train_data = value['train-mse']
                test_data = value['test-mse']
                time_data = value['time']
                train_mse_list.append(train_data)
                test_mse_list.append(test_data)
                time_list.append(time_data)
            list_train_mse_list.append(train_mse_list)
            list_test_mse_list.append(test_mse_list)
            list_time_list.append(time_list)
            # The typical rule I apply here is that as the batch size increases, we need to increase learning rate at the same time. 0.01 is the optimize learning
            # rate for 1, I suppose lr = previous lr + 0.02 for next iteration
            self.lr = self.lr +0.06


        for index, the_list in enumerate(list_time_list):
            plt.plot(the_list, list_train_mse_list[index])
        plt.ylabel('Train-MSE')
        plt.xlabel('time')
        plt.legend(batch_size_list)
        plt.ylim(top= 7)
        plt.ylim(bottom = 0)
        plt.show()

        # plt.plot(the_list, list_test_mse_list[index])


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    # setting the seed for deterministic behavior
    np.random.seed(args.seed)
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
