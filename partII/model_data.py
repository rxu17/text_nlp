# -*- coding: utf-8 -*-
''' Name: model_data.py
    Description: Runs pre-processing then classification models 
        on input invoice data in order to predict the Condition variable

    Arguments:
        train_filename (str): Training data filename
        test_filename (str): Test data filename
        filedir (str): Input file directory for both test and training data
        steps (str, Optional): Available steps are in preprocess_data.preprocess_steps()
        text_cols (str, Optional): list of columns to use as predictors
        target (str, Optionla): target variable name (to predict)
        
    How to Run: 
        python model_data.py <train_filename> 
                             <test_filename> 
                             <filedir> 
                             <steps> 
                             <text_cols>
                             <target>
        
    Example)
        python model_data.py input.csv 
                             test.csv 
                             "Some_filepath" 
                             1,2,3,4,5,6,7,8,9 
                             count_type, count_id
                             weather

    Contributors: rxu17
'''

import re
import sys
import preprocess_data as prep
import pandas as pd

# helper ML functions
from sklearn.model_selection import(
    GridSearchCV, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import(
    CountVectorizer, TfidfTransformer,
    TfidfVectorizer
)

# ML classification models to try
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import (
    SGDClassifier,  LogisticRegression
)
import xgboost as xgb

# scoring
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, make_scorer
)


def get_model(model_type :str) -> object:
    ''' Get the model object based on given model type
    '''
    models = {"decision": DecisionTreeClassifier,
             "naive": MultinomialNB,
             "sdg":SGDClassifier,
             "logistic": LogisticRegression}
    return(models[model_type])


def get_model_param_settings(model_type : str) -> dict:
    ''' Gets the range of hyperparams for each model type
        to have GridSearch iterate over
    '''
    if model_type == "decision":
        hyperparams = {"max_depth":2,
                    "min_samples_split":1,
                    "min_samples_leaf":1}
        hyperparams = {"max_depth":[1, 3, 5, 10, 15],
                    "min_samples_split":[0.5, 3, 5, 10, 15],
                    "min_samples_leaf":[1, 3, 5, 10, 15]}
    elif model_type == "xgb":
        hyperparams = {"max_depth":[1, 5, 10],
                    "n_estimators":[10, 50, 100],
                    "max_leaves":[1, 5, 10],
                    "tree_method":["hist", "approx"]}
    elif model_type == "naive":
        hyperparams = {"alpha":0.001}
        hyperparams = {"alpha":[0.001, 0.01, 0.1, 1]}
        hyperparams = [{"clf":[get_model(model_type)()],
                        "clf__alpha":[0.001, 0.01, 0.1, 1]}]
    elif model_type == "logistic":
        # choosing solvers
        # For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
        # For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
        hyperparams = {"penalty":'elasticnet',
                      "C":0.001,
                      "solver":"saga",
                      "l1_ratio":0.4}
        hyperparams = {"penalty":['elasticnet'],
                       "C":[0.001, 0.01, 0.1, 0.5],
                       "solver":["saga"],
                       "l1_ratio":[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}
        hyperparams = [{"clf":[get_model(model_type)()],
                        'clf__penalty': ['l1', 'l2'],
                        'clf__C': [1.0, 0.5, 0.1],
                        'clf__solver': ['liblinear']}]
    elif model_type == "sdg":
        hyperparams = {"loss":'hinge', 
                       "penalty":'l2',
                       "alpha":1e-3,
                       "max_iter":5, 
                       "tol":None}
        hyperparams = {"loss":['hinge'], 
                       "penalty":['l2'],
                       "alpha":[1e-3, 1e-2, 1e-1, 1],
                       "max_iter":[5], 
                       "tol":[None]}
        hyperparams = [{"clf":[get_model(model_type)()],
                        'clf__penalty': ['l1', 'l2'],
                        'clf__alpha':[1e-3, 1e-2, 1e-1, 1],
                        'clf__max_iter': [5]}]
    elif model_type == "svm":
        hyperparams = [{"clf":[get_model(model_type)()],
                        'clf__kernel': ['linear', 'rbf'], 
                        'clf__C': [1, 2, 3, 4, 5, 6]}]
    else:
        raise ValueError("model_type passed doesn't exist")
    return(hyperparams)


def find_best_model_params(model : object, hyperparams : dict,
                           X_train : pd.DataFrame,
                           y_train : pd.DataFrame,
                           cv : int = 10) -> dict:
    """Gridsearch function to find best model params
        given a dict of hyperparams

    Args:
        model (object): Model object that we are using as predictor
        X_train (pd.DataFrame): training data with only predictors
        y_train (pd.DataFrame): training data with only target var
        hyperparams (dict): contains each model's parameters and 
                    list of values for each param to iterate on
        cv (int, optional): parameter for grid search. Defaults to 6.

    Returns:
        dict: of hyperparams and their best values
    """
    scoring = {'AUC':'roc_auc', 
               'Accuracy' : make_scorer(accuracy_score)}
    gs = GridSearchCV(estimator = model,
                      param_grid=hyperparams,
                      scoring = 'accuracy',
                      n_jobs = -1, #use all processors
                      cv = cv)
    best_model = gs.fit(X = X_train, y = y_train)
    return(best_model)


def run_model(classifier, X_train, y_train, X_test, y_test, hyperparams= None):
    """Run models on the best grid serched hyperparams, outputs
        classification scores and returns the predicted values

    Args:
        classifier : Model object that we are using as predictor
        X_train : training data with only predictors
        y_train : training data with only target var
        X_test : test data with only predictors
        y_test : test data with only target var
        hyperparams (_type_, optional): contains each model's parameters and 
                    list of values for each param to iterate on. Defaults to None.
    """
    model_pipe = Pipeline([('vect', CountVectorizer()),
                           ('tfidf', TfidfTransformer()),
                           ('clf', MultinomialNB())])
    model_pipe = find_best_model_params(model = model_pipe,
                                        hyperparams=hyperparams,
                                        X_train = X_train,
                                        y_train = y_train)
    y_pred = model_pipe.predict(X_test)
    if y_test is not None:
        # Measuring accuracy on predicted results
        print(classification_report(y_test, y_pred))
        print(confusion_matrix(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))
    return(y_pred)


def main():
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    filedir = sys.argv[3]
    prep_steps = sys.argv[4].split(",")
    text_col = sys.argv[5]
    target = sys.argv[6]
    prep_results_saved = bool(int(sys.argv[7]))
    model_type = sys.argv[8]
    
    # running models on validation data
    X_train, X_test, y_train, y_test = prep.preprocess_w_nlp(
        train_filename, test_filename, filedir, 
        prep_steps, text_col, target, is_test = False,
        results_saved = prep_results_saved)
    model_inputs = {
        "X_train": X_train, 
        "y_train": y_train, 
        "X_test": X_test, 
        "y_test":y_test
    }
    for model_type in ['logistic', 'naive', 'sdg']:
        model = get_model(model_type)
        y_pred = run_model(model, **model_inputs, hyperparams = get_model_param_settings(model_type))
        
    # predicting for actual test data
    train_df, test_df = prep.preprocess_w_nlp(
        train_filename, test_filename, filedir, 
        prep_steps, text_col, target, is_test = True,
        results_saved = prep_results_saved)

    model_test_inputs = {
        "X_train": X_train, 
        "y_train": y_train, 
        "X_test": X_test, 
        "y_test": None
    }

if __name__ == "__main__":
    main()