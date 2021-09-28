import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import acquire

# ------------------------ Wrangle MVP ----------------------- #
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
    df = fix_MVP_zillow_columns(df)
    # Drop outliers
    cols = ['Home.Value','Baths','Beds','Finished.Area']
    df = remove_outliers(df, 1.5, cols)
    # Add engineered features
    df = south_coastline(df)
    df = cool_areas(df)
    # Split data
    train, validate, test = split_data(df)
    # Isolate target
    X_train, y_train,\
    X_validate, y_validate,\
    X_test, y_test = isolate_zillow_target('Prediction.Error', train, validate, test)
    # Create exploration dataframe before encoding
    X_train_exp = X_train.copy()
    X_train_exp = pd.concat([X_train_exp, y_train], axis=1)
    # Encode columns that need it
    map1 = {True:0, False:1}
    X_train['is_coastal'] = X_train['is_coastal'].map(map1)
    X_validate['is_coastal'] = X_validate['is_coastal'].map(map1)
    X_test['is_coastal'] = X_test['is_coastal'].map(map1)
    X_train['cool_places'] = X_train['cool_places'].map(map1)
    X_validate['cool_places'] = X_validate['cool_places'].map(map1)
    X_test['cool_places'] = X_test['cool_places'].map(map1)
    # Scale data
    X_train, X_validate, X_test = scale_MVP_zillow(X_train, X_validate, X_test)
    # Add X_train_scaled columns to X_train
    X_train_exp[['Home.Value.Scaled',
                'Baths.Scaled',
                'Beds.Scaled',
                'Finished.Area.Scaled',
                'Is.Coastal.Scaled',
                'Is.Cool.Place.Scaled']] = X_train
    
    # Return everything
    return df, X_train_exp, X_train, y_train, X_validate, y_validate, X_test, y_test

def fix_MVP_zillow_columns(df):
    """ Return a dataframe of a few fixed Zillow columns for the MVP """
    # Fix column dtypes
    df['parcelid'] = df.parcelid.astype('str')
    df['fips'] = df.fips.astype('int').astype('str')
    df['latitude'] = df.latitude.astype('int')
    df['longitude'] = df.longitude.astype('int')
    df['taxvaluedollarcnt'] = df.taxvaluedollarcnt.astype('int')
    df['calculatedfinishedsquarefeet'] = df.calculatedfinishedsquarefeet.astype('int')
    # Fix column names
    df = df.rename(columns={'parcelid':'Parcel.ID',
                            'fips':'County',
                            'latitude':'Latitude',
                            'longitude':'Longitude',
                            'taxvaluedollarcnt':'Home.Value',
                            'logerror':'Prediction.Error',
                            'bathroomcnt':'Baths',
                            'bedroomcnt':'Beds',
                            'calculatedfinishedsquarefeet':'Finished.Area'})
    # Map County numbers to names
    map1 = {'6037':'LA', '6059':'Orange', '6111':'Ventura'}
    df['County'] = df.County.map(map1)

    return df

def scale_MVP_zillow(X_train, X_validate, X_test):
    """ Applies StandardScaler to Zillow split/isolated data """
    scaler = StandardScaler()
    columns_to_scale = ['Home.Value','Baths','Beds',
                        'Finished.Area', 'is_coastal', 'cool_places']
    X_train_scaled = scaler.fit_transform(X_train[columns_to_scale])
    X_validate_scaled = scaler.transform(X_validate[columns_to_scale])
    X_test_scaled = scaler.transform(X_test[columns_to_scale])

    return X_train_scaled, X_validate_scaled, X_test_scaled

# ----------------------- General Functions ----------------------- #

def remove_outliers(df, k, col_list):
    ''' Remove outliers from a list of columns in a dataframe 
        and returns that dataframe
    '''
    for col in col_list:
        q1, q3 = df[col].quantile([.25, .75])  # Get quartiles
        iqr = q3 - q1   # Calculate interquartile range
        upper_bound = q3 + k * iqr   # Get upper bound
        lower_bound = q1 - k * iqr   # Get lower bound
        # Apply IQR
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df

def south_coastline(df):
    """ Marks some lat/longs that I identified as coastal areas """
    # Weaken precision to 4 decimal places (divide by 1 million, round) and
    # Ensure all values are even (divide by 2 before rounding, multiply by 2 after rounding)
    # (I chose even because it gave best results)
    df['relaxed_lat'] = 2 * round(df['Latitude'] / (2 * 1000000), 4)
    df['relaxed_long'] = 2 * round(df['Longitude'] / (2 * 1000000), 4)
    # Create two dictionaries for lat:long and long:lat, iterating
    latlong_dict = dict(df.groupby('relaxed_lat').Longitude.min())
    longlat_dict = dict(df.groupby('relaxed_long').Latitude.min())
    # Reference lat:long dict and the northern bound for southern coastal
    df['is_coastal_horizontal'] = ((df['relaxed_lat'] <= 33.9) & 
                                    (df['relaxed_lat'].apply(lambda x: 
                                     latlong_dict.get(x)) == df['Longitude']))
    # Reference long:lat dict and the western bound for southern coastal
    df['is_coastal_vertical'] = ((df['relaxed_long'] >= -118.40) & 
                                (df['relaxed_long'].apply(lambda x: 
                                 longlat_dict.get(x)) == df['Latitude']))
    # One column showing True/False if in both identified groups
    df['is_coastal'] = df['is_coastal_vertical'] & df['is_coastal_horizontal']
    # Drop the approaches
    df = df.drop(columns=['relaxed_lat','relaxed_long',
                            'is_coastal_horizontal',
                            'is_coastal_vertical',])
    
    return df

def cool_areas(df):
    """ Marks some lat/long areas I identified in Explore """
    # Create True/False for lat/long in individual areas
    df['square'] = ((df.Latitude > 33850000) & (df.Latitude < 34000000) & 
                    (df.Longitude > -118200000) & (df.Longitude < -118000000))
    df['rectangle'] = ((df.Latitude > 33950000) & (df.Latitude < 34070000) & 
                        (df.Longitude > -118150000) & (df.Longitude < -117650000))
    df['diagonal_0'] = ((df.Latitude > 34150000) & (df.Latitude < 34200000) & 
                        (df.Longitude > -118650000) & (df.Longitude < -118560000))
    df['diagonal_1'] = ((df.Latitude > 34190000) & (df.Latitude < 34210000) & 
                        (df.Longitude > -118650000) & (df.Longitude < -118480000))
    df['diagonal_2'] = ((df.Latitude > 34220000) & (df.Latitude < 34350000) & 
                        (df.Longitude > -118650000) & (df.Longitude < -118340000))
    # One column showing True or False for lat/longs in any specified area
    df['cool_places'] = (df['is_coastal'] | df['square'] | df['rectangle'] | 
                     df['diagonal_0'] | df['diagonal_1'] | df['diagonal_2'])
    # Drop the individual areas
    df = df.drop(columns=['square','rectangle',
                 'diagonal_0','diagonal_1','diagonal_2'])

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