import pandas as pd
import os

from env import host, password, username

def get_db_url(db_name, username=username, hostname=host, password=password):
    """ Build an SQL query using env credentials """
    return f'mysql+pymysql://{username}:{password}@{hostname}/{db_name}'

def get_clustering_zillow():
    """ Acquire each table from Codeup database 'zillow' from 2017 with 
        one entry per parcel (latest transaction date) and no nulls in lat/long where
        every entry is Single Family Residential property"""
    # Build query
    url = get_db_url(db_name='zillow')
    query = """ 
                SELECT *, properties_2017.id as property_id, predictions_2017.id as transaction_id
                FROM properties_2017
                JOIN predictions_2017 USING(parcelid)
                LEFT JOIN airconditioningtype USING(airconditioningtypeid)
                LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
                LEFT JOIN buildingclasstype USING(buildingclasstypeid)
                LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid)
                LEFT JOIN propertylandusetype USING(propertylandusetypeid)
                LEFT JOIN storytype USING(storytypeid)
                LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)
                LEFT JOIN unique_properties USING(parcelid)
                WHERE (transactiondate BETWEEN '2017-01-01' AND '2017-12-31') 
	                AND (latitude IS NOT NULL) 
	                AND (longitude IS NOT NULL)
                    AND propertylandusedesc = 'Single Family Residential'
            """
    # Run query
    df1 = pd.read_sql(query, url)
    # Create separate indexing df with one entry per parcel
    df2 = pd.DataFrame(df1.groupby('parcelid').transactiondate.max()).reset_index()
    # Join the two dataframes
    df = pd.merge(df2, df1, on=['parcelid','transactiondate'])
    # Drop duplicate-named 'id' columns
    df = df.drop(columns='id')

    return df

def pull_clustering_zillow():
    """ Returns a dataframe of Zillow data, stores Zillow data in directory if not found there"""
    if not os.path.isfile('fresh_zillow.csv'):
        df = get_clustering_zillow()
        df.to_csv('fresh_zillow.csv')
    df = pd.read_csv('fresh_zillow.csv', index_col=0)
    return df

def MVP_pull_clustering_zillow():
    """ Return dataframe of columns used for MVP, before preparing columns """
    if not os.path.isfile('MVP_zillow.csv'):
        url = get_db_url(db_name='zillow')
        query = """
                    SELECT propertylandusetypeid, parcelid, fips, latitude, longitude, taxvaluedollarcnt,
                        logerror, bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet
                    FROM properties_2017
                    JOIN predictions_2017 USING(parcelid)
                    WHERE (transactiondate BETWEEN '2017-01-01' AND '2017-12-31') 
	                    AND (latitude IS NOT NULL) 
	                    AND (longitude IS NOT NULL)
                        AND propertylandusetypeid = 261
                """
        df = pd.read_sql(query, url).drop(columns='propertylandusetypeid')
        df.to_csv('MVP_zillow.csv')
    df = pd.read_csv('MVP_zillow.csv', index_col=0)
    return df