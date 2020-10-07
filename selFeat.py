import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y %H:%M')
    # extract specific date, time
    df['actual_date'], df['time'] = df['date'].dt.normalize(), df['date'].dt.time
    # extract month
    df['month'] = pd.DatetimeIndex(df['actual_date']).month
    # extract day
    df['day'] = pd.DatetimeIndex(df['actual_date']).day
    # extract hour
    df['hour'] = df.date.dt.hour
    # extract minutes
    df['min'] = df.date.dt.minute

    df = df.drop(columns=['date', 'actual_date', 'time'])

    return df


def get_correlation_matrix(df):
    return df.corr()


def plot_df_heatmap(df):
    plt.figure(figsize=(30, 20))
    ax = plt.subplot(111)
    return sns.heatmap(df, ax=ax, annot=True, fmt='.1g', vmin=-1, vmax=1, center=0, cmap='coolwarm')


def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data 
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    df = df[['month', 'day',
             'hour', 'min', 'Visibility',
             'Windspeed', 'Press_mm_hg', 'RH_out',
             'T7', 'RH_2', 'lights', ]]
    return df


def preprocess_data(trainDF, testDF):
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data 
    testDF : pandas dataframe
        Test data 
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    scale = MinMaxScaler(feature_range=(0, 1))
    trainDF = scale.fit_transform(trainDF)
    testDF = scale.fit_transform(testDF)
    trainDF = pd.DataFrame(trainDF)
    testDF = pd.DataFrame(testDF)
    trainDF.columns = ['month', 'day', 'hour',
                       'min', 'Visibility', 'Windspeed',
                       'Press_mm_hg', 'RH_out',
                       'T7', 'RH_2',
                       'lights']
    testDF.columns = ['month', 'day', 'hour',
                      'min', 'Visibility', 'Windspeed',
                      'Press_mm_hg', 'RH_out',
                      'T7', 'RH_2',
                      'lights']
    
    return trainDF, testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.trainFile)
    xTest = pd.read_csv(args.testFile)
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv(args.outTrain, index=False)
    xTestTr.to_csv(args.outTest, index=False)


if __name__ == "__main__":
    main()
