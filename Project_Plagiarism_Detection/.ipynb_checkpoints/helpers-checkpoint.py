import re
import pandas as pd
import operator 

# Add 'datatype' column that indicates if the record is original wiki answer as 0, training data 1, test data 2, onto 
# the dataframe - uses stratified random sampling (with seed) to sample by task & plagiarism amount 

# Use function to label datatype for training 1 or test 2 
def create_datatype(df, train_value, test_value, datatype_var, compare_dfcolumn, operator_of_compare, value_of_compare,
                    sampling_number, sampling_seed):
    # Subsets dataframe by condition relating to statement built from:
    # 'compare_dfcolumn' 'operator_of_compare' 'value_of_compare'
    df_subset = df[operator_of_compare(df[compare_dfcolumn], value_of_compare)]
    df_subset = df_subset.drop(columns = [datatype_var])
    
    # Prints counts by task and compare_dfcolumn for subset df
    #print("\nCounts by Task & " + compare_dfcolumn + ":\n", df_subset.groupby(['Task', compare_dfcolumn]).size().reset_index(name="Counts") )
    
    # Sets all datatype to value for training for df_subset
    df_subset.loc[:, datatype_var] = train_value
    
    # Performs stratified random sample of subset dataframe to create new df with subset values 
    df_sampled = df_subset.groupby(['Task', compare_dfcolumn], group_keys=False).apply(lambda x: x.sample(min(len(x), sampling_number), random_state = sampling_seed))
    df_sampled = df_sampled.drop(columns = [datatype_var])
    # Sets all datatype to value for test_value for df_sampled
    df_sampled.loc[:, datatype_var] = test_value
    
    # Prints counts by compare_dfcolumn for selected sample
    #print("\nCounts by "+ compare_dfcolumn + ":\n", df_sampled.groupby([compare_dfcolumn]).size().reset_index(name="Counts") )
    #print("\nSampled DF:\n",df_sampled)
    
    # Labels all datatype_var column as train_value which will be overwritten to 
    # test_value in next for loop for all test cases chosen with stratified sample
    for index in df_sampled.index: 
        # Labels all datatype_var columns with test_value for straified test sample
        df_subset.loc[index, datatype_var] = test_value

    #print("\nSubset DF:\n",df_subset)
    # Adds test_value and train_value for all relevant data in main dataframe
    for index in df_subset.index:
        # Labels all datatype_var columns in df with train_value/test_value based upon 
        # stratified test sample and subset of df
        df.loc[index, datatype_var] = df_subset.loc[index, datatype_var]

    # returns nothing because dataframe df already altered 
    
def train_test_dataframe(clean_df, random_seed=100):
    
    new_df = clean_df.copy()

    # Initialize datatype as 0 initially for all records - after function 0 will remain only for original wiki answers
    new_df.loc[:,'Datatype'] = 0

    # Creates test & training datatypes for plagiarized answers (1,2,3)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.gt, 0, 1, random_seed)

    # Creates test & training datatypes for NON-plagiarized answers (0)
    create_datatype(new_df, 1, 2, 'Datatype', 'Category', operator.eq, 0, 2, random_seed)
    
    # creating a dictionary of categorical:numerical mappings for plagiarsm categories
    mapping = {0:'orig', 1:'train', 2:'test'} 

    # traversing through dataframe and replacing categorical data
    new_df.Datatype = [mapping[item] for item in new_df.Datatype] 

    return new_df


# helper function for pre-processing text given a file
def process_file(file):
    # put text in all lower case letters 
    all_text = file.read().lower()

    # remove all non-alphanumeric chars
    all_text = re.sub(r"[^a-zA-Z0-9]", " ", all_text)
    # remove newlines/tabs, etc. so it's easier to match phrases, later
    all_text = re.sub(r"\t", " ", all_text)
    all_text = re.sub(r"\n", " ", all_text)
    all_text = re.sub("  ", " ", all_text)
    all_text = re.sub("   ", " ", all_text)
    
    return all_text


def create_text_column(df, file_directory='data/'):
    '''Reads in the files, listed in a df and returns that df with an additional column, `Text`. 
       :param df: A dataframe of file information including a column for `File`
       :param file_directory: the main directory where files are stored
       :return: A dataframe with processed text '''
   
    # create copy to modify
    text_df = df.copy()
    
    # store processed text
    text = []
    
    # for each file (row) in the df, read in the file 
    for row_i in df.index:
        filename = df.iloc[row_i]['File']
        #print(filename)
        file_path = file_directory + filename
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:

            # standardize text using helper function
            file_text = process_file(file)
            # append processed text to list
            text.append(file_text)
    
    # add column to the copied dataframe
    text_df['Text'] = text
    
    return text_df
