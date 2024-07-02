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

# Remember Data First # For all, we are going to use halving grid search

# Load Data
def main():
    train_remember_x = joblib.load("features/train_remember_x.pkl")
    train_remember_y = joblib.load("features/train_remember_y.pkl")
    test_remember_x = joblib.load("features/test_remember_x.pkl")
    test_remember_y = joblib.load("features/test_remember_y.pkl")
    combined_remember_x_auto = joblib.load("features/combined_remember_x_auto.pkl")
    remember_y_auto = joblib.load("features/remember_y_auto.pkl")
    column_names_remember = joblib.load("features/column_names_remember.pkl")

    # Naive Bayes
    params_nb = {'var_smoothing': [1e-8, 1e-9, 1e-10]}
    params_svm = {'C': [0.1, 1, 10, 100],
              'gamma': ['scale', 'auto'],
              'kernel': ['linear', 'poly', 'rbf']}
    params_lr = {'penalty': ['l1', 'l2', 'none'],
             'C': [0.1, 1, 10],
             'solver': ['saga'],
             'tol': [0.01, 0.001, 0.0001],
             'max_iter': [200, 500]}
    
    params_rf = {'n_estimators': [50, 100, 250],
             'max_depth': [None, 5, 10],
             'max_features':['auto', 'sqrt'],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4],
             'bootstrap': [True, False]}
    
    params_xgb = {'gamma':[0.1, 0.5],
              'learning_rate': [0.1, 0.5],
              'max_depth': [5, 7, 10],
              'n_estimators': [50, 100]}

    gnb_rememb = GaussianNB()
    gnb_remember_hgs = HalvingGridSearchCV(gnb_rememb, params_nb, scoring="f1", n_jobs = -1, cv=3, verbose = 3)
    start_time = time.time()
    gnb_remember_hgs.fit(train_remember_x, train_remember_y)
    end_time = time.time()
    print("Time taken to train Naive Bayes for Remember: ", end_time - start_time)
    joblib.dump(gnb_remember_hgs, "models/gnb_remember_gs.joblib")
   
    print("Best parameters for Naive Bayes for Remember: ", gnb_remember_hgs.best_params_)
    pred_remember_y_gnb = gnb_remember_hgs.predict(test_remember_x)
    pred_remember_y_gnb_auto = gnb_remember_hgs.predict(combined_remember_x_auto)
    print("Accuracy for Naive Bayes for Remember", accuracy_score(test_remember_y, pred_remember_y_gnb))
    performancePrinter(test_remember_y, pred_remember_y_gnb)
    print("Accuracy for Naive Bayes for Remember with Auto: ", accuracy_score(remember_y_auto, pred_remember_y_gnb_auto))
    performancePrinter(remember_y_auto, pred_remember_y_gnb_auto)

    #Support Vector Machine
    SVC_rememb = SVC()
    svc_remember_hgs = HalvingGridSearchCV(SVC_rememb, params_svm,score='f1', n_jobs = -1, cv=3, verbose = 3)
    start_time = time.time()
    svc_remember_hgs.fit(train_remember_x, train_remember_y)
    end_time = time.time()
    print("Time taken to train SVM for Remember: ", end_time - start_time)
    joblib.dump(svc_remember_hgs, "models/svc_remember_gs.joblib")
    print("Best parameters for SVM for Remember: ", svc_remember_hgs.best_params_)
    pred_remember_y_svc = svc_remember_hgs.predict(test_remember_x)
    pred_remember_y_svc_auto = svc_remember_hgs.predict(combined_remember_x_auto)
    print("Accuracy for SVM for Remember", accuracy_score(test_remember_y, pred_remember_y_svc))
    performancePrinter(test_remember_y, pred_remember_y_svc)
    print("Accuracy for SVM for Remember with Auto: ", accuracy_score(remember_y_auto, pred_remember_y_svc_auto))
    performancePrinter(remember_y_auto, pred_remember_y_svc_auto)


    #Logistic Regression
    lr_rememb = LogisticRegression()
    lr_remember_hgs = HalvingGridSearchCV(lr_rememb, params_lr, scoring='f1', n_jobs = -1, cv=3, verbose = 3)
    start_time = time.time()
    lr_remember_hgs.fit(train_remember_x, train_remember_y)
    end_time = time.time()
    print("Time taken to train Logistic Regression for Remember: ", end_time - start_time)
    joblib.dump(lr_remember_hgs, "models/lr_remember_gs.joblib")
    print("Best parameters for Logistic Regression for Remember: ", lr_remember_hgs.best_params_)
    pred_remember_y_lr = lr_remember_hgs.predict(test_remember_x)
    pred_remember_y_lr_auto = lr_remember_hgs.predict(combined_remember_x_auto)
    print("Accuracy for Logistic Regression for Remember", accuracy_score(test_remember_y, pred_remember_y_lr))
    performancePrinter(test_remember_y, pred_remember_y_lr)
    print("Accuracy for Logistic Regression for Remember with Auto: ", accuracy_score(remember_y_auto, pred_remember_y_lr_auto))
    performancePrinter(remember_y_auto, pred_remember_y_lr_auto)


    # Random Forest
    rf_rememb = RandomForestClassifier()
    rf_remember_hgs = HalvingGridSearchCV(rf_rememb, params_rf, scoring='f1', n_jobs = -1, cv=3, verbose = 3)
    start_time = time.time()
    rf_remember_hgs.fit(train_remember_x, train_remember_y)
    end_time = time.time()
    print("Time taken to train Random Forest for Remember: ", end_time - start_time)
    joblib.dump(rf_remember_hgs, "models/rf_remember_gs.joblib")
    print("Best parameters for Random Forest for Remember: ", rf_remember_hgs.best_params_)
    pred_remember_y_rf = rf_remember_hgs.predict(test_remember_x)
    pred_remember_y_rf_auto = rf_remember_hgs.predict(combined_remember_x_auto)
    print("Accuracy for Random Forest for Remember", accuracy_score(test_remember_y, pred_remember_y_rf))
    performancePrinter(test_remember_y, pred_remember_y_rf)
    print("Accuracy for Random Forest for Remember with Auto: ", accuracy_score(remember_y_auto, pred_remember_y_rf_auto))
    performancePrinter(remember_y_auto, pred_remember_y_rf_auto)

    # XGBoost
    xgb_rememb = XGBClassifier()
    xgb_remember_hgs = HalvingGridSearchCV(xgb_rememb, params_xgb, scoring='f1', n_jobs = -1, cv=3, verbose = 3)
    start_time = time.time()
    xgb_remember_hgs.fit(train_remember_x, train_remember_y)
    end_time = time.time()
    print("Time taken to train XGBoost for Remember: ", end_time - start_time)
    joblib.dump(xgb_remember_hgs, "models/xgb_remember_gs.joblib")
    print("Best parameters for XGBoost for Remember: ", xgb_remember_hgs.best_params_)
    pred_remember_y_xgb = xgb_remember_hgs.predict(test_remember_x)
    pred_remember_y_xgb_auto = xgb_remember_hgs.predict(combined_remember_x_auto)
    print("Accuracy for XGBoost for Remember", accuracy_score(test_remember_y, pred_remember_y_xgb))
    performancePrinter(test_remember_y, pred_remember_y_xgb)
    print("Accuracy for XGBoost for Remember with Auto: ", accuracy_score(remember_y_auto, pred_remember_y_xgb_auto))
    performancePrinter(remember_y_auto, pred_remember_y_xgb_auto)




if __name__ == "__main__":
    main()