import argparse
import joblib
from collections import Counter
import xlrd
import pandas as pd
import openpyxl
import numpy as np 
from nltk import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, cohen_kappa_score, f1_score
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, AutoModelForSequenceClassification, EarlyStoppingCallback
from datasets import load_metric, list_metrics
import time
from skopt import BayesSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import textstat
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt

def performancePrinter(test_y, pred_y):
    # performance printer
    print("Accuracy Score -> ", accuracy_score(test_y, pred_y))
    print("Kappa Score -> ", cohen_kappa_score(test_y, pred_y))
    print("ROC AUC Score -> ", roc_auc_score(test_y, pred_y))
    print("F1 Score -> ", f1_score(test_y, pred_y))
    print("Classification report -> \n", classification_report(test_y, pred_y))

def load_data(task):
    train_x = joblib.load(f"features/train_{task.lower()}_x.pkl")
    train_y = joblib.load(f"features/train_{task.lower()}_y.pkl")
    test_x = joblib.load(f"features/test_{task.lower()}_x.pkl")
    test_y = joblib.load(f"features/test_{task.lower()}_y.pkl")
    combined_x_auto = joblib.load(f"features/combined_{task.lower()}_x_auto.pkl")
    y_auto = joblib.load(f"features/{task.lower()}_y_auto.pkl")
    return train_x, train_y, test_x, test_y, combined_x_auto, y_auto

def run_model(model, params, train_x, train_y, test_x, test_y, combined_x_auto, y_auto, task, model_name):
    model_hgs = HalvingGridSearchCV(model, params, scoring='f1', n_jobs=-1, cv=3, verbose=3)
    start_time = time.time()
    model_hgs.fit(train_x, train_y)
    end_time = time.time()
    print(f"Time taken to train {model_name} for {task}: ", end_time - start_time)
    joblib.dump(model_hgs, f"models/{model_name.lower()}_{task.lower()}_gs.joblib")
    print(f"Best parameters for {model_name} for {task}: ", model_hgs.best_params_)
    pred_y = model_hgs.predict(test_x)
    pred_y_auto = model_hgs.predict(combined_x_auto)
    print(f"Accuracy for {model_name} for {task}", accuracy_score(test_y, pred_y))
    performancePrinter(test_y, pred_y)
    print(f"Accuracy for {model_name} for {task} with Auto: ", accuracy_score(y_auto, pred_y_auto))
    performancePrinter(y_auto, pred_y_auto)

def main(tasks):
    for task in tasks:
        print(f"Running models for {task} task...")
        train_x, train_y, test_x, test_y, combined_x_auto, y_auto = load_data(task)
        
        # Naive Bayes
        params_nb = {'var_smoothing': [1e-8, 1e-9, 1e-10]}
        run_model(GaussianNB(), params_nb, train_x, train_y, test_x, test_y, combined_x_auto, y_auto, task, "Naive Bayes")

        # Support Vector Machine
        params_svm = {'C': [0.1, 1, 10, 100],
                      'gamma': ['scale', 'auto'],
                      'kernel': ['linear', 'poly', 'rbf']}
        run_model(SVC(), params_svm, train_x, train_y, test_x, test_y, combined_x_auto, y_auto, task, "SVM")

        # Logistic Regression
        params_lr = {'penalty': ['l1', 'l2', 'none'],
                     'C': [0.1, 1, 10],
                     'solver': ['saga'],
                     'tol': [0.01, 0.001, 0.0001],
                     'max_iter': [200, 500]}
        run_model(LogisticRegression(), params_lr, train_x, train_y, test_x, test_y, combined_x_auto, y_auto, task, "Logistic Regression")

        # Random Forest
        params_rf = {'n_estimators': [50, 100, 250],
                     'max_depth': [None, 5, 10],
                     'max_features':['auto', 'sqrt'],
                     'min_samples_split': [2, 5, 10],
                     'min_samples_leaf': [1, 2, 4],
                     'bootstrap': [True, False]}
        run_model(RandomForestClassifier(), params_rf, train_x, train_y, test_x, test_y, combined_x_auto, y_auto, task, "Random Forest")

        # XGBoost
        params_xgb = {'gamma':[0.1, 0.5],
                      'learning_rate': [0.1, 0.5],
                      'max_depth': [5, 7, 10],
                      'n_estimators': [50, 100]}
        run_model(XGBClassifier(), params_xgb, train_x, train_y, test_x, test_y, combined_x_auto, y_auto, task, "XGBoost")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run machine learning models on different tasks")
    parser.add_argument('tasks', nargs='+', choices=['Remember', 'Understand', 'Analyze', 'Apply'], help="The tasks to run the models for")
    args = parser.parse_args()
    main(args.tasks)
