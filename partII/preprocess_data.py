# -*- coding: utf-8 -*-
''' Name: preprocess_data.py
    Description: Does pre-processing in the form of standarizing, 
    tokenizing, other NLP cleaning functions on scraped invoice data
    to prep it for modeling

    How to Run: 
        In another script:
            import preprocess_data as prep

            df = [some dataframe to preprocess]
            prep.remove_names(df) # individual functions
            OR
            prep.preprocess_w_nlp(
                train_filename, test_filename, filedir, steps, text_cols, model)
    Contributors: rxu17
'''


# import required libraries
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

# for NLP processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# word embedding methods
# bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# word to vector
import smart_open
smart_open.open = smart_open.smart_open
from gensim.models import Word2Vec

# helper ML functions
from sklearn.model_selection import train_test_split

# set logging style
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')

# setting global variables for ease of access by functions
VECTORIZE_METHOD = "word2vec"
VALIDATION_SIZE = 0.2


def download_nltk_functions_banks() -> None:
    """ Downloads nltk data
    """
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')
    logging.info("All nltk data downloaded!")


def preprocess_steps() -> dict:
    ''' Returns: 
            dict: dictionary of each preprocessing step
            attached to the function it's associated with
    '''
    return({1:remove_special_char,
            2:remove_punctuation,
            3:strip_whitespace,
            4:convert_to_lower_case,
            5:remove_stop_words,
            6:tokenize_data,
            7:stem,
            8:lemmatize,
            9:vectorize})


def read_and_format(file_name : str, 
                    input_dir : str) -> pd.DataFrame:
    """ Reads and does some custom formatting for input data 
        prior to other NLP processing

    Args:
        file_name (str): file name of file to be read in
        input_dir (str): directory path of file to be read in

    Returns:
        pd.DataFrame: formatted dataset
    """
    filepath = "{}/{}".format(input_dir, file_name)
    assert os.path.isfile(filepath),\
        "This is not a valid filepath :{}".format(filepath)

    df = pd.read_csv(filepath)
    df = custom_processing(df)
    logging.info("Dataset read in.")
    return(df)


def custom_processing(input_df : pd.DataFrame, 
                      col : str = None) -> pd.DataFrame:
    """Initial custom pre-processing for data.
       Fills NA
       Combines NotesOne and NotesTwo columns"""
    # combine notes one and two
    df = input_df.copy()
    for col in ['NotesOne', 'NotesTwo']:
        df[col] = fillNA(df, col)
    df['NotesFull'] = df['NotesOne'] + " " + df['NotesTwo']
    logging.info("Early dataset pre-processing done.")
    return(df)


def fillNA(input_df : pd.DataFrame,
           col: str,
           fillNA_method = "constant",
           fillNA_value = "") -> pd.Series:
    """Fills NAs in selected col with selected fillNA_method and/or
        fillNA_values. Currently, only the constant fillNA_method
        is available for use.

    Args:
        input_df (pd.DataFrame): input data to fill NAs for
        col (str): column in input_df to fill NAs for
        fillNA_method (str, optional): Defaults to "constant".
        fillNA_value (str, optional): Defaults to "".
    """
    logging.info("Fill in NAs with {} method".format(fillNA_method))
    if fillNA_method == "constant":
        return(input_df[col].fillna(fillNA_value))
    else:
        return(input_df[col])
    

def remove_special_char(input_df : pd.DataFrame,
                        col : str) -> pd.Series:
    """ Removes special characters for a given dataset and specific var
    """
    logging.info("Removing special characters ...")
    return(input_df[col].str.replace(r'[^a-zA-Z ]', regex=True, repl=' '))


def remove_punctuation(input_df : pd.DataFrame,
                       col : str) -> pd.Series:
    """ Removes punctuation for a given dataset and specific variable
    """
    logging.info("Removing punctuation ...")
    return(input_df[col].str.replace(r'[^\w\s]', regex = True, repl=' '))


def strip_whitespace(input_df : pd.DataFrame,
                     col : str) -> pd.Series:
    """ Removes trailing and leading whitespace for a given dataset
        and specific variable
    """
    logging.info("Stripping trailing and leading whitespace ...")
    return(input_df[col].str.strip())


def convert_to_lower_case(input_df : pd.DataFrame,
                          col : str) -> pd.Series:
    """convert text column in input_df to lower case
    """
    logging.info("Convert to lower case ...")
    return(input_df[col].str.lower())


def remove_stop_words(input_df : pd.DataFrame, col : str) -> pd.Series:
    ''' Remove what are defined as stop words (e.g: that, and)
        in the english language from the given input data and text column
    '''
    words = stopwords.words('english')
    logging.info("Removing stop words ...")
    return(input_df[col].apply(lambda x: ' '.join([
        word for word in x.split() if word not in words
        ])))


def tokenize_data(input_df : pd.DataFrame, col : str) -> pd.Series:
    ''' Converts the text column in the given input data into
        a list of words
    '''
    logging.info("Tokenizing data ...")
    return(input_df[col].apply(lambda x: word_tokenize(x)))


def stem(input_df : pd.DataFrame, col : str) -> pd.Series:
    ''' Converts each word in the tokenized text column of input data
        into its stem form (e.g: training -> train)
    '''
    logging.info("Stemming data ...")
    ps = PorterStemmer() 
    return(input_df[col].apply(
        lambda x: [ps.stem(w) for w in x]))


def lemmatize(input_df : pd.DataFrame, col : str) -> pd.Series:
    ''' Lemmatizes each word in the tokenized text column of input data
        (e.g: training -> train)
    '''
    logging.info("Lemmatizing data ...")
    def _get_wordnet_pos(word):
        '''POS tagger to help lemmatize function'''
        tag = nltk.pos_tag([word])[0][1][0].upper()
        pos_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return pos_dict.get(tag, wordnet.NOUN)

    lemmatizer = WordNetLemmatizer()
    return(input_df[col].apply(
       lambda x: [lemmatizer.lemmatize(w, _get_wordnet_pos(w)) for w in x]))


def vectorize(train_df, test_df, col, method_type : str = VECTORIZE_METHOD):
    ''' Converts the given training data's text column from text to 
        numeric based on the selected method
        
        method_type (str) : Allowed values currently are 
                            ['word2vec', 'count', 'tfid']
    '''
    if method_type not in ['word2vec', 'count', 'tfid']:
        raise ValueError("{} must be in  ['word2vec', 'count', 'tfid']".format(method_type))
    logging.info("Vectorizing data using {} method ...".format(method_type))
    if method_type == "word2vec":
        return(word2vec(train_df, test_df, col))
    elif method_type == "count":
        vect = CountVectorizer()
    elif method_type == "tfid":
        vect = TfidfVectorizer(use_idf=True)
    
    # re-combined tokenized words
    X_train_cnt = vect.fit_transform(
        train_df[col].apply(lambda x : " ".join(x)).to_numpy()).toarray()
    X_test_cnt = vect.transform(
        test_df[col].apply(lambda x : " ".join(x)).to_numpy()).toarray()
    return(pd.DataFrame(data=X_train_cnt, columns = vect.get_feature_names()),
           pd.DataFrame(data=X_test_cnt, columns = vect.get_feature_names()))


def word2vec(train_df, test_df, col, word2vec_model : str = None):
    ''' Word2vec model where each word is emdedded as a series 
        of numbers
    '''
    tokens = train_df[col].values
    model = Word2Vec(tokens, size=300, min_count=1, workers=4)
    logging.info("\n Training the word2vec model...\n")
    # reducing the epochs will decrease the computation time
    model.train(tokens, total_examples=len(tokens), epochs=4000)

    #building Word2Vec model
    def _rowwise_word2vec(row, model):
        ''' Take the mean of each word in the word2vec model's values
            and adds the means up 
        '''
        total = 0
        for word in row:
            try:
                total += np.mean(model[word])
            except:
                total += 0
        return(total)

    return(train_df[col].apply(lambda x :_rowwise_word2vec(x, model)),
           test_df[col].apply(lambda x :_rowwise_word2vec(x, model)))


def preprocess_w_nlp(train_filename : str, 
                     test_filename : str,
                     filedir : str, steps : list=preprocess_steps().keys(), 
                     text_cols : list = None, 
                     is_test : bool = False) -> pd.DataFrame:
    """Preprocesses data with functions from user input using preprocess_steps()

    Args:
        train_filename (str): Training data filename
        test_filename (str):  Test data filename
        filedir (str): Input file directory for both test and training data
        steps (list, optional): Available steps are in preprocess_steps(). 
                Defaults to preprocess_steps().keys().
        text_cols (list, optional): list of columns to use as predictors. 
                Defaults to None.
        is_test (bool, optional) : Whether we're pre-processing our testing data 
                or validation data (is_test = False). Defaults to False.

    Returns:
        pd.DataFrame: pre-processed train_df and test_df
    """
    train_df = read_and_format(train_filename, filedir)
    test_df = read_and_format(test_filename, filedir)
    
    # split into training and validation
    train_df, validation_data = \
        train_test_split(train_df, test_size=VALIDATION_SIZE, random_state=6)
    prep_steps = preprocess_steps()
    
    # "test" data is validation if specified by is_test
    test_df = validation_data.copy() if not is_test else test_df
    for step in steps:
        logging.info("Currently on step: {}".format(step))
        # perform standardization on entire dataset
        for text_col in text_cols:
            if prep_steps[int(step)].__name__ == "vectorize":
                train_df[text_col], test_df[text_col] = prep_steps[int(step)](train_df, test_df, text_col)
            else:
                train_df[text_col] = prep_steps[int(step)](train_df, text_col)
                test_df[text_col] = prep_steps[int(step)](test_df, text_col)
    
    return(train_df, test_df)