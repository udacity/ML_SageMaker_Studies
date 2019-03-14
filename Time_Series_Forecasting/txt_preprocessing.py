# This file was used to process the initial, raw text: household_power_consumption.txt
import pandas as pd


## 1. Raw data processing

# The 'household_power_consumption.txt' file has the following attributes:
#    * Each data point has a date and time (hour) of recording
#    * The data points are separated by semicolons (;)
#    * Some values are 'nan' or '?', and we'll treat these both as `NaN` values when making a DataFrame

# A helper function to read the file in and create a DataFrame, indexed by 'Date Time'
def create_df(text_file, sep=';', na_values=['nan','?']):
    '''Reads in a text file and converts it to a dataframe, indexed by 'Date Time'.'''
    
    df = None
    
    # check that the file is the expected text file
    expected_file='household_power_consumption.txt'
    if(text_file != expected_file):
        print('Unexpected file: '+str(text_file))
        return df
    
    # read in the text file
    # each data point is separated by a semicolon
    df = pd.read_csv('household_power_consumption.txt', sep=sep, 
                     parse_dates={'Date-Time' : ['Date', 'Time']}, infer_datetime_format=True, 
                     low_memory=False, na_values=na_values, index_col='Date-Time') # indexed by Date-Time

    return df

## 2. Managing `NaN` values

# This DataFrame does include some data points that have missing values. 
# So far, we've mainly been dropping these values, but there are other ways to handle `NaN` values. 
# One technique is to fill the missing values with the *mean* values from a column, 
# this way the added value is likely to be realistic.

# A helper function to fill NaN values with a column average
def fill_nan_with_mean(df):
    '''Fills NaN values in a given dataframe with the average values in a column.
       This technique works well for filling missing, hourly values 
       that will later be averaged into energy stats over a day (24hrs).'''
    
    # filling nan with mean value of any columns
    num_cols = len(list(df.columns.values))
    for col in range(num_cols):        
        df.iloc[:,col]=df.iloc[:,col].fillna(df.iloc[:,col].mean())
        
    return df
    
    