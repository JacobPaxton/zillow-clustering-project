import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import acquire

def wrangle_zillow_MVP():
    """ Acquires MVP columns for Zillow,
        Prepares columns,
        Splits data,
        Isolates target,
         and returns dataframe """
    # Acquire data
    df = acquire.MVP_pull_clustering_zillow()
    # Drop nulls
    df = df.dropna()
    # Fix columns
    df = fix_MVP_zillow_column_dtypes(df)
    # Drop outliers
    col_list = ['taxvaluedollarcnt','bathroomcnt','bedroomcnt',
                        'calculatedfinishedsquarefeet']
    df = remove_outliers(df, 1.5, col_list)
    # Split data
    train, validate, test = split_data(df)
    # Isolate target
    X_train, y_train,\
    X_validate, y_validate,\
    X_test, y_test = isolate_zillow_target('logerror', train, validate, test)
    # Scale data
    X_train_scaled, X_validate_scaled, X_test_scaled = scale_MVP_zillow(X_train, X_validate, X_test)
    # Add X_train_scaled columns to X_train
    X_train_exp = X_train.copy()
    X_train_exp[['worth_scaled',
                'baths_scaled',
                'beds_scaled',
                'finishedarea_scaled']] = X_train_scaled
    
    # Return everything
    return df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test

def fix_MVP_zillow_column_dtypes(df):
    """ Return a dataframe of a few fixed Zillow columns for the MVP"""
    # Fix column dtypes
    df['parcelid'] = df.parcelid.astype('str')
    df['fips'] = df.fips.astype('int').astype('str')
    df['latitude'] = df.latitude.astype('int')
    df['longitude'] = df.longitude.astype('int')
    df['taxvaluedollarcnt'] = df.taxvaluedollarcnt.astype('int')
    df['calculatedfinishedsquarefeet'] = df.calculatedfinishedsquarefeet.astype('int')

    return df

def remove_outliers(df, k, col_list):
    ''' Remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # Get quartiles
        iqr = q3 - q1   # Calculate interquartile range
        upper_bound = q3 + k * iqr   # Get upper bound
        lower_bound = q1 - k * iqr   # Get lower bound
        # Apply IQR
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def split_data(df):
    """ Standard 60 / 20 / 20 data split for exploration and modeling"""
    train_validate, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=0.25, random_state=123)

    return train, validate, test

def isolate_zillow_target(target, train, validate, test):
    """ Isolates target column from train, validate, test splits of Zillow,
        then returns all dataframes """
    X_train, y_train = train.drop(columns=target), train[[target]]
    X_validate, y_validate = validate.drop(columns=target), validate[[target]]
    X_test, y_test = test.drop(columns=target), test[[target]]

    return X_train, y_train, X_validate, y_validate, X_test, y_test

def scale_MVP_zillow(X_train, X_validate, X_test):
    """ Applies StandardScaler to Zillow split/isolated data """
    scaler = StandardScaler()
    columns_to_scale = ['taxvaluedollarcnt','bathroomcnt','bedroomcnt',
                        'calculatedfinishedsquarefeet']
    X_train_scaled = scaler.fit_transform(X_train[columns_to_scale])
    X_validate_scaled = scaler.transform(X_validate[columns_to_scale])
    X_test_scaled = scaler.transform(X_test[columns_to_scale])
    return X_train_scaled, X_validate_scaled, X_test_scaled

