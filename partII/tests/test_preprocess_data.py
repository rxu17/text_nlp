# -*- coding: utf-8 -*-
import os
import pytest
import numpy as np
import pandas as pd
from unittest import mock
from trupanion_test.partII import preprocess_data as prep


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


def test_fillNA(mock_datasets):
    prep.fillNA(
        mock_datasets['df_text_pass'], 'text', "test fillNA")[7] == "test fillNA"

def test_remove_special_char_pass(mock_datasets):
    assert prep.remove_special_char(
        mock_datasets['df_text_pass'], 'text')[0] == "Test  for  special char "

def test_remove_special_char_fail():
    pass

def test_remove_punctuation_pass(mock_datasets):
    assert prep.remove_punctuation(
        mock_datasets['df_text_pass'], 'text')[1] == "Test for  punctuation "

def test_remove_punctuation_fail():
    pass

def test_strip_whitespace_pass(mock_datasets):
    assert prep.strip_whitespace(
        mock_datasets['df_text_pass'], 'text')[2] == "Test for whitespace"

def test_strip_whitespace_fail():
    pass

def test_convert_to_lower_case_pass(mock_datasets):
    assert prep.convert_to_lower_case(
        mock_datasets['df_text_pass'], 'text')[3] == "test for lowercase"


def test_convert_to_lower_case_fail():
    pass

def test_remove_stop_words_pass(mock_datasets):
    assert prep.remove_stop_words(
        mock_datasets['df_text_pass'], 'text')[4] == "Test stop words"

def test_remove_stop_words_fail():
    pass

def test_tokenize_pass(mock_datasets):
    assert prep.tokenize_data(
        mock_datasets['df_text_pass'], 'text')[5] == ["Test", "for", "tokenize"]

def test_tokenize_fail():
    pass

def test_stem_pass(mock_datasets):
    assert prep.stem(
        mock_datasets['df_words_pass'], 'words')[0] == ["test", "for", "stem"]

def test_stem_fail():
    pass

def test_lemmatize_pass(mock_datasets):
    assert prep.lemmatize(
        mock_datasets['df_words_pass'], 'words')[1] == ["test", "for", "lemmatization"]

def test_lemmatize_fail():
    pass

def test_vectorize_count_pass(mock_datasets):
    prep.vectorize(mock_datasets['df_text_pass'], 'text')
    
    
def test_vectorize_tify_pass(mock_datasets):
    prep.vectorize(mock_datasets['df_text_pass'], 'text')

def test_vectorize_fail():
    pass