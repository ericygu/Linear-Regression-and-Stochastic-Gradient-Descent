import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        start = time.time()

        trainStats = {}

        # Configure xTrain xTest
        ones_xTrain = np.ones((len(xTrain), 1))
        ones_xTest = np.ones((len(xTest), 1))
        xTrain = np.concatenate((ones_xTrain, xTrain), axis=1)
        xTest = np.concatenate((ones_xTest, xTest), axis=1)

        # Linear Regression Formula
        LinearRegression.beta = np.linalg.inv(xTrain.T.dot(xTrain)).dot(xTrain.T).dot(yTrain)

        # Calculate MSE

        train_mse = self.mse(xTrain, yTrain)
        test_mse = self.mse(xTest, yTest)

        end = time.time()
        timeElapsed = end - start

        trainStats[0] = {}
        trainStats[0]['time'] = timeElapsed
        trainStats[0]['train-mse'] = train_mse
        trainStats[0]['test-mse'] = test_mse

        return trainStats


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

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
