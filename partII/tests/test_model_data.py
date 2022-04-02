# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np
import pandas as pd
from unittest import mock
from trupanion_test.partII import model_data as model

@pytest.fixture
def mock_datasets():
    ''' Generate dataframe
        examples for each of the test functions
    '''
    df_text_pass = pd.DataFrame({
        "id":[1,2,3,4,5,6,7,8],
        "text":[
        'Test ^!&#$for !<>special char*',
        'Test for ?punctuation![]',
        ' Test for whitespace  ',
        'Test FoR LoWeRCaSe',
        'Test for stop words how an are what',
        'Test for tokenize',
        'Test for vectorize',
        np.NaN]
    })
    df_words_pass = pd.DataFrame({
        'id':[1,2],
        'words':[['Test', 'for', 'stemming'],
        ["testing", "for", "lemmatization"]]
    })
    return({'df_text_pass':df_text_pass,
            'df_words_pass':df_words_pass})
    
    
def test_find_best_model_params(model : object, X_train : pd.DataFrame,
                           y_train : pd.DataFrame, hyperparams : dict,
                           cv : int = 6)


def test_run_model(classifier, X_train, y_train, X_test, y_test, hyperparams= None)