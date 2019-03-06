from unittest.mock import MagicMock, patch
import sklearn.naive_bayes
import numpy as np
import pandas as pd
import re

# test csv file
TEST_CSV = 'data/test_info.csv'

class AssertTest(object):
    '''Defines general test behavior.'''
    def __init__(self, params):
        self.assert_param_message = '\n'.join([str(k) + ': ' + str(v) + '' for k, v in params.items()])
    
    def test(self, assert_condition, assert_message):
        assert assert_condition, assert_message + '\n\nUnit Test Function Parameters\n' + self.assert_param_message

def _print_success_message():
    print('Tests Passed!')

# test clean_dataframe
def test_numerical_df(numerical_dataframe):
    
    # test result
    transformed_df = numerical_dataframe(TEST_CSV)
                                
    # Check type is a DataFrame
    assert isinstance(transformed_df, pd.DataFrame), 'Returned type is {}.'.format(type(transformed_df))
    
    # check columns
    column_names = list(transformed_df)
    assert 'File' in column_names, 'No File column, found.'
    assert 'Task' in column_names, 'No Task column, found.'
    assert 'Category' in column_names, 'No Category column, found.'
    assert 'Class' in column_names, 'No Class column, found.'
                                       
    # check conversion values
    assert transformed_df.loc[0, 'Category'] == 1, '`heavy` plagiarism mapping test, failed.'
    assert transformed_df.loc[2, 'Category'] == 0, '`non` plagiarism mapping test, failed.'
    assert transformed_df.loc[30, 'Category'] == 3, '`cut` plagiarism mapping test, failed.'
    assert transformed_df.loc[5, 'Category'] == 2, '`light` plagiarism mapping test, failed.'
    assert transformed_df.loc[37, 'Category'] == -1, 'original file mapping test, failed; should have a Category = -1.'
    assert transformed_df.loc[41, 'Category'] == -1, 'original file mapping test, failed; should have a Category = -1.'
    
    _print_success_message()


def test_containment(complete_df, containment_fn):
    
    # check basic format and value 
    # for n = 1 and just the fifth file
    test_val = containment_fn(complete_df, 1, 'g0pA_taske.txt')
    
    assert isinstance(test_val, float), 'Returned type is {}.'.format(type(test_val))
    assert test_val<=1.0, 'It appears that the value is not normalized; expected a value <=1, got: '+str(test_val)
    
    # known vals for first few files
    filenames = ['g0pA_taska.txt', 'g0pA_taskb.txt', 'g0pA_taskc.txt', 'g0pA_taskd.txt']
    ngram_1 = [0.39814814814814814, 1.0, 0.86936936936936937, 0.5935828877005348]
    ngram_3 = [0.0093457943925233638, 0.96410256410256412, 0.61363636363636365, 0.15675675675675677]
    
    # results for comparison
    results_1gram = []
    results_3gram = []
    
    for i in range(4):
        val_1 = containment_fn(complete_df, 1, filenames[i])
        val_3 = containment_fn(complete_df, 3, filenames[i])
        results_1gram.append(val_1)
        results_3gram.append(val_3)
        
    # check correct results
    assert all(np.isclose(results_1gram, ngram_1, rtol=1e-04)), \
    'n=1 calculations are incorrect. Double check the intersection calculation.'
    # check correct results
    assert all(np.isclose(results_3gram, ngram_3, rtol=1e-04)), \
    'n=3 calculations are incorrect.'
    
    _print_success_message()
    
def test_lcs(df, lcs_word):
    
    test_index = 10 # file 10
    
    # get answer file text
    answer_text = df.loc[test_index, 'Text'] 
    
    # get text for orig file
    # find the associated task type (one character, a-e)
    task = df.loc[test_index, 'Task']
    # we know that source texts have Class = -1
    orig_rows = df[(df['Class'] == -1)]
    orig_row = orig_rows[(orig_rows['Task'] == task)]
    source_text = orig_row['Text'].values[0]
    
    # calculate LCS
    test_val = lcs_word(answer_text, source_text)
    
    # check type
    assert isinstance(test_val, float), 'Returned type is {}.'.format(type(test_val))
    assert test_val<=1.0, 'It appears that the value is not normalized; expected a value <=1, got: '+str(test_val)
    
    # known vals for first few files
    lcs_vals = [0.1917808219178082, 0.8207547169811321, 0.8464912280701754, 0.3160621761658031, 0.24257425742574257]
    
    # results for comparison
    results = []
    
    for i in range(5):
        # get answer and source text
        answer_text = df.loc[i, 'Text'] 
        task = df.loc[i, 'Task']
        # we know that source texts have Class = -1
        orig_rows = df[(df['Class'] == -1)]
        orig_row = orig_rows[(orig_rows['Task'] == task)]
        source_text = orig_row['Text'].values[0]
        # calc lcs
        val = lcs_word(answer_text, source_text)
        results.append(val)
        
    # check correct results
    assert all(np.isclose(results, lcs_vals, rtol=1e-05)), 'LCS calculations are incorrect.'
    
    _print_success_message()
    
def test_data_split(train_x, train_y, test_x, test_y):
        
    # check types
    assert isinstance(train_x, np.ndarray),\
        'train_x is not an array, instead got type: {}'.format(type(train_x))
    assert isinstance(train_y, np.ndarray),\
        'train_y is not an array, instead got type: {}'.format(type(train_y))
    assert isinstance(test_x, np.ndarray),\
        'test_x is not an array, instead got type: {}'.format(type(test_x))
    assert isinstance(test_y, np.ndarray),\
        'test_y is not an array, instead got type: {}'.format(type(test_y))
        
    # should hold all 95 submission files
    assert len(train_x) + len(test_x) == 95, \
        'Unexpected amount of train + test data. Expecting 95 answer text files, got ' +str(len(train_x) + len(test_x))
    assert len(test_x) > 1, \
        'Unexpected amount of test data. There should be multiple test files.'
        
    # check shape
    assert train_x.shape[1]==2, \
        'train_x should have as many columns as selected features, got: {}'.format(train_x.shape[1])
    assert len(train_y.shape)==1, \
        'train_y should be a 1D array, got shape: {}'.format(train_y.shape)
    
    _print_success_message()
    
    
        