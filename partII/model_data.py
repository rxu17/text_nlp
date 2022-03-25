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

import sys
import preprocess_data as prep
import pandas as pd

# helper ML functions
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# ML classification models to try
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import xgboost as xgb

# scoring
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

def find_best_model_params(model : object, X_train : pd.DataFrame,
                           y_train : pd.DataFrame, hyperparams : dict,
                           cv : int = 6) -> dict:
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
    if isinstance(X_train, pd.DataFrame):
        train_df = X_train.to_numpy()
    else:
        train_df = X_train.copy()
    search = GridSearchCV(estimator = model,
                          param_grid=hyperparams,
                          return_train_score=True,
                          cv = cv).fit(X = train_df, 
                                      y = y_train.to_numpy())
    return(search.best_params_)


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
    best_params = find_best_model_params(model = classifier(),
                                        X_train = X_train,
                                        y_train = y_train, 
                                        hyperparams = hyperparams)
    model = classifier(**best_params).fit(X_train,y_train)
    y_pred = model.predict(X_test)
    if y_test is not None:
        # Measuring accuracy on predicted results
        print(metrics.classification_report(y_test, y_pred))
        print(metrics.confusion_matrix(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))
    return(y_pred)


def main():
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    filedir = sys.argv[3]
    prep_steps = sys.argv[4].split(",")
    text_cols = sys.argv[5].split(",")
    target = sys.argv[6]
    
    # running models on validation data
    train_df, test_df = prep.preprocess_w_nlp(
        train_filename, test_filename, filedir, prep_steps, text_cols, is_test = False)
    
    hyperparams = {"max_depth":[1, 3, 5, 10, 15],
                   "min_samples_split":[0.5, 3, 5, 10, 15],
                   "min_samples_leaf":[1, 3, 5, 10, 15]}
    run_model(DecisionTreeClassifier, X_train=train_df[text_cols], 
             y_train=train_df[target], 
             X_test = test_df[text_cols], 
             y_test = test_df[target], hyperparams = hyperparams)
    
    hyperparams = {"max_depth":[1, 5, 10],
                   "n_estimators":[10, 50, 100],
                   "max_leaves":[1, 5, 10],
                   "tree_method":["hist", "approx"]}
    run_model(xgb.XGBClassifier, X_train=train_df[text_cols], 
             y_train=train_df[target],
             X_test = test_df[text_cols], 
             y_test = test_df[target], hyperparams = hyperparams)
    
    # predicting for actual test data
    train_df, test_df = prep.preprocess_w_nlp(
        train_filename, test_filename, filedir, prep_steps, text_cols, is_test = True)
    hyperparams = {"max_depth":[1, 3, 5, 10, 15],
                   "min_samples_split":[0.5, 3, 5, 10, 15],
                   "min_samples_leaf":[1, 3, 5, 10, 15]}
    test_df[target] = run_model(DecisionTreeClassifier, X_train=train_df[text_cols], 
                            y_train=train_df[target], 
                            X_test = test_df[text_cols], 
                            y_test = None, hyperparams = hyperparams)


if __name__ == "__main__":
    main()