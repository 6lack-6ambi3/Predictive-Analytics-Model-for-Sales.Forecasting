import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Function to read data from a CSV file and return a DataFrame
def read_data(data_path):
    '''
    The function reads the data in the CSV format from the path mentioned 
    and returns a DataFrame.
    Input (String): the data path
    Output (DataFrame): Read Pandas DataFrame
    '''
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(e)
    else:
        return df

# Function to split the data into train and test DataFrames
def split_data(data, size, randomstate):
    '''
    The function splits the data into train and test DataFrames
    data (DataFrame): The input DataFrame
    '''
    try:
        X = data.drop(columns=['Item_Outlet_Sales'])
        y = data['Item_Outlet_Sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=randomstate)
    except Exception as e:
        print(e)
    else:
        return X_train, X_test, y_train, y_test
