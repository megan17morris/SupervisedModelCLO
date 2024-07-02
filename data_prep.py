

    
def main():
    #clean_data(auto_questions='data/Autoquestionbank.xlsx', auto_LWIC='data/lwicauto.xlsx', oriiginal_questions='data/sample_full.csv', original_LWIC='data/lwicOriginal.xlsx')
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import textstat
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as plt
import time
from skopt import BayesSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV


from sklearn.svm import SVC

def clean_data( auto_questions, auto_LWIC,oriiginal_questions='None', original_LWIC='None'):
    auto_LWIC = pd.read_excel(auto_LWIC)
    auto_questions = pd.read_excel(auto_questions)
    auto_questions = auto_questions.rename(columns={"question":"Learning_outcome"})
    mapping = {
    'Remembering': 'Remember',
    'Understanding': 'Understand',
    'Applying': 'Apply',
    'Analyzing': 'Analyze',
    'Evaluating': 'Evaluate',
    'Creating': 'Create'}
    #Create new columns based on the mapping
    for old_value, new_column in mapping.items():
        auto_questions[new_column] = auto_questions['objective'].apply(lambda x: 1.0 if x == old_value else 0.0)
    auto_questions = auto_questions.drop(columns=["Number", "objective", "Bloom's Level Judged", "strategy", "scenario", "options", "correct_answer", "explanation", "textbook_section","model", "analysis"])
    auto_questions['one_hot_encoded']=list(auto_questions[auto_questions.columns[1:]].values)
    auto_questions['Learning_outcome'] = auto_questions['Learning_outcome'].str.lower()
    textual_data_auto = auto_questions['Learning_outcome'].tolist()

    auto_LWIC = auto_LWIC.drop(columns=["Number", "Segment", "objective", "Bloom's Level Judged", "strategy", "scenario", "options", "correct_answer", "explanation", "textbook_section","model", "analysis"])
    auto_LWIC = auto_questions.join(auto_LWIC)
    auto_LWIC = auto_LWIC.drop(columns=["question"])
    print(auto_LWIC.columns)
    auto_LWIC = auto_LWIC.fillna(0.0)
    auto_LWIC.to_excel("data/cleaned_auto_LWIC.xlsx", index=False)
    data = pd.read_csv(oriiginal_questions, delimiter=',', skipinitialspace=True, quotechar='"')
    data.fillna({'Remember': 0, 'Understand': 0, 'Apply': 0, 'Analyze': 0, 'Evaluate': 0, 'Create':0}, inplace=True)
    data['one_hot_encoded'] = list(data[data.columns[1:]].values)
    data['Learning_outcome'] = data['Learning_outcome'].str.lower()
    textual_data = data['Learning_outcome'].tolist()
    lengths = []
    for text in textual_data:
        lengths.append(len(word_tokenize(text)))
    LIWC_data = pd.read_excel(original_LWIC)
    LIWC_data["Learning_outcome"] = LIWC_data["Learning_outcome"].str.lower()
    LIWC_data=LIWC_data.drop(columns=["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create", "Segment"])
    data = data.join(LIWC_data, rsuffix='_LIWC')
    data = data.drop(columns=["Learning_outcome_LIWC"])
    print(data.columns)
    data = data.fillna(0.0)
    data.to_excel("data/cleaned_original_LWIC.xlsx", index=False)

class EncodeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def generateX(data_x, test_x, textual_column_index, start_index_LIWC, end_index_LIWC):
    # generating ML features based on previous literature
    # Can try to run using less features for storage
    column_names = []
    print("Getting Unigram...")
    uni_cv = CountVectorizer(stop_words='english', ngram_range=(1, 1), max_features=1000)
    unigram = uni_cv.fit_transform(data_x[:, textual_column_index])
    unigram = unigram.toarray()
    unigram_test = uni_cv.transform(test_x[:,textual_column_index]).toarray()
    temp = uni_cv.get_feature_names_out().tolist()
    column_names += ["uni_"+name for name in temp]
    print("Getting Bigram...")
    bi_cv = CountVectorizer(stop_words='english', ngram_range=(2, 2), max_features=1000)
    bigram = bi_cv.fit_transform(data_x[:, textual_column_index])
    bigram = bigram.toarray()
    bigram_test = bi_cv.transform(test_x[:, textual_column_index]).toarray()
    temp = bi_cv.get_feature_names_out().tolist()
    column_names += ["bi_"+name for name in temp]
    print("Getting Tfidf...")
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_features=1000)
    t = tfidf.fit_transform(data_x[:, textual_column_index])
    t = t.toarray()
    t_test = tfidf.transform(test_x[:, textual_column_index]).toarray()
    temp = tfidf.get_feature_names_out().tolist()
    column_names += ["tfidf_"+name for name in temp]
    print("Getting ARI...")
    ari = [textstat.automated_readability_index(text) for text in data_x[:, textual_column_index]]
    ari_test = [textstat.automated_readability_index(text) for text in test_x[:, textual_column_index]]
    column_names.append("ari")
    combined_data_x = []
    combined_test_x = []
    print("Combining...")
    for i in range(len(data_x)):
        combined_data_x.append(unigram[i].tolist()
                              + bigram[i].tolist()
                              + t[i].tolist()
                              + [ari[i]]
                              + data_x[i, start_index_LIWC:end_index_LIWC].tolist())
    for i in range(len(test_x)):
        combined_test_x.append(unigram_test[i].tolist()
                              + bigram_test[i].tolist()
                              + t_test[i].tolist()
                              + [ari_test[i]]
                              + test_x[i, start_index_LIWC:end_index_LIWC].tolist())
    print("Generated feature shape is", np.array(combined_data_x).shape)
    print("Generated test feature is", np.array(combined_test_x).shape)
    return combined_data_x, column_names, combined_test_x, uni_cv, bi_cv, tfidf

def transformX(test_x,textual_column_index,start_index_LIWC, end_index_LIWC,uni_cv, bi_cv, tfidf):
    column_names = []
    print("Getting Unigram...")
    unigram_test = uni_cv.transform(test_x[:,textual_column_index]).toarray()
    temp = uni_cv.get_feature_names_out().tolist()
    column_names += ["uni_"+name for name in temp]
    print("Getting Bigram...")
    bigram_test = bi_cv.transform(test_x[:, textual_column_index]).toarray()
    temp = bi_cv.get_feature_names_out().tolist()
    column_names += ["bi_"+name for name in temp]
    print("Getting Tfidf...")
    t_test = tfidf.transform(test_x[:, textual_column_index]).toarray()
    temp = tfidf.get_feature_names_out().tolist()
    column_names += ["tfidf_"+name for name in temp]
    print("Getting ARI...")
    
    ari_test = [textstat.automated_readability_index(text) for text in test_x[:, textual_column_index]]
    column_names.append("ari")
    
    combined_test_x = []
    print("Combining...")
 
    for i in range(len(test_x)):
        combined_test_x.append(unigram_test[i].tolist()
                              + bigram_test[i].tolist()
                              + t_test[i].tolist()
                              + [ari_test[i]]
                              + test_x[i, start_index_LIWC:end_index_LIWC].tolist())
    print("Generated test feature is", np.array(combined_test_x).shape)

    return combined_test_x


    
def performancePrinter(test_y, pred_y):
    # performance printer
    print("Accuracy Score -> ", accuracy_score(test_y, pred_y))
    print("Kappa Score -> ", cohen_kappa_score(test_y, pred_y))
    print("ROC AUC Score -> ", roc_auc_score(test_y, pred_y))
    print("F1 Score -> ", f1_score(test_y, pred_y))
    print("Classification report -> \n", classification_report(test_y, pred_y))
    
    
def createBERT(dir_name, X, Y, test_X, test_Y, batch_size=64, nepochs=3, patience=10):
    # function to fine-tune BERT with given data and print out performance on the testing set
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_cache=False)
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, use_cache=False)
    training_args = TrainingArguments(
        output_dir=dir_name,          # output directory
        num_train_epochs=nepochs,              # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=batch_size,   # batch size for evaluation
        warmup_steps=5,                # number of warmup steps for learning rate scheduler
        weight_decay=0.05,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        metric_for_best_model="f1",
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=10,
        load_best_model_at_end=True,
        save_total_limit=3
    )
    train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=0.2, random_state=666, stratify=Y)

    train_encoded = tokenizer(train_x, truncation=True, padding=True, max_length=55)
    val_encoded = tokenizer(val_x, truncation=True, padding=True, max_length=55)
    test_encoded = tokenizer(test_X, truncation=True, padding=True, max_length=55)

    train_set = EncodeDataset(train_encoded, train_y)
    val_set = EncodeDataset(val_encoded, val_y)
    test_set = EncodeDataset(test_encoded, test_Y)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_set, eval_dataset=val_set, compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])
    print("Started training model for column", dir_name)
    trainer.train()
    trainer.save_model()
    print("Training Completed. Started testing...")
    predicted = trainer.predict(test_set)
    predicted_result = np.argmax(predicted.predictions, axis=-1)
    print("Accuracy Score -> ", accuracy_score(test_Y, predicted_result))
    print("Kappa Score -> ", cohen_kappa_score(test_Y, predicted_result))
    print("ROC AUC Score -> ", roc_auc_score(test_Y, predicted_result))
    print("F1 Score -> ", f1_score(test_Y, predicted_result))
    print("Classification report -> \n", classification_report(test_Y, predicted_result))
    return trainer




    
def main():
    #clean_data(auto_questions='data/Autoquestionbank.xlsx', auto_LWIC='data/lwicauto.xlsx', oriiginal_questions='data/sample_full.csv', original_LWIC='data/lwicOriginal.xlsx')
    data = pd.read_excel("data/cleaned_original_LWIC.xlsx")
    lwicauto = pd.read_excel("data/cleaned_auto_LWIC.xlsx")
    metric = load_metric("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="macro")
    
    remember_bert = None
    understand_bert = None
    apply_bert = None
    analyze_bert = None
    evaluate_bert = None
    create_bert = None
    # Define Parameters 
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
    
    # # Split original data into train and test
    # split_train_x, split_test_x, split_train_y, split_test_y = train_test_split(data.drop(columns=list(data.columns[1:8])), data[data.columns[1:7]], test_size=0.2, random_state=666)
    # # Separate Autodata into x and y
    # lwic_x = lwicauto.drop(columns=list(lwicauto.columns[1:8]))
    # lwic_y = lwicauto[lwicauto.columns[1:7]]
    # # Generate Remember data for x and y for auto
    # remember_x_auto, remember_y_auto = lwic_x.to_numpy(), lwic_y['Remember'].astype('long').to_numpy()
    # # Generate Remember data for x and y for original
    # remember_x, remember_y = split_train_x.to_numpy(), split_train_y['Remember'].astype('long').to_numpy()

    # # Create feature vectors for original 
    # combined_remember_x, column_names_remember, test_remember_x, uni_cv, bi_cv,tfifd = generateX(remember_x, split_test_x.to_numpy(), 0, 1, 119)
    # train_remember_x = combined_remember_x
    # train_remember_y = remember_y
    # test_remember_y = split_test_y['Remember'].astype('long').to_numpy()
    # # Create feature vectors for auto
    # combined_remember_x_auto = transformX(remember_x_auto,0,1,119, uni_cv, bi_cv, tfifd)
    # # Add file names for the feature vector
    # column_names_remember += data.columns[8:].tolist()

    # #Features to save: train_remember_x, train_remember_y, test_remember_x, test_remember_y, combined_remember_x_auto,  column_names_remember, remember_y_auto
    # joblib.dump(train_remember_x, "features/train_remember_x.pkl")
    # joblib.dump(train_remember_y, "features/train_remember_y.pkl")
    # joblib.dump(test_remember_x, "features/test_remember_x.pkl")
    # joblib.dump(test_remember_y, "features/test_remember_y.pkl")
    # joblib.dump(combined_remember_x_auto, "features/combined_remember_x_auto.pkl")
    # joblib.dump(remember_y_auto, "features/remember_y_auto.pkl")
    # joblib.dump(column_names_remember, "features/column_names_remember.pkl")


    # Split original data into train and test
    split_train_x, split_test_x, split_train_y, split_test_y = train_test_split(data.drop(columns=list(data.columns[1:8])), data[data.columns[1:7]], test_size=0.2, random_state=666)
    # Separate Autodata into x and y
    lwic_x = lwicauto.drop(columns=list(lwicauto.columns[1:8]))
    lwic_y = lwicauto[lwicauto.columns[1:7]]
    # Generate Understnand data for x and y for auto
    apply_x_auto, apply_y_auto = lwic_x.to_numpy(), lwic_y['Apply'].astype('long').to_numpy()
    # Generate Understand data for x and y for original
    apply_x, apply_y = split_train_x.to_numpy(), split_train_y['Apply'].astype('long').to_numpy()

    # Create feature vectors for original 
    combined_apply_x, column_names_apply, test_apply_x, uni_cv, bi_cv,tfifd = generateX(apply_x, split_test_x.to_numpy(), 0, 1, 119)
    train_apply_x = combined_apply_x
    train_apply_y = apply_y
    test_apply_y = split_test_y['Apply'].astype('long').to_numpy()
    # Create feature vectors for auto
    combined_apply_x_auto = transformX(apply_x_auto,0,1,119, uni_cv, bi_cv, tfifd)
    # Add file names for the feature vector
    column_names_apply += data.columns[8:].tolist()

    #Features to save: train_remember_x, train_remember_y, test_remember_x, test_remember_y, combined_remember_x_auto,  column_names_remember, remember_y_auto
    joblib.dump(train_apply_x, "features/train_apply_x.pkl")
    joblib.dump(train_apply_y, "features/train_apply_y.pkl")
    joblib.dump(test_apply_x, "features/test_apply_x.pkl")
    joblib.dump(test_apply_y, "features/test_apply_y.pkl")
    joblib.dump(combined_apply_x_auto, "features/combined_apply_x_auto.pkl")
    joblib.dump(apply_y_auto, "features/apply_y_auto.pkl")
    joblib.dump(column_names_apply, "features/column_names_apply.pkl")

    # Load the saved features
    train_apply_x = joblib.load("features/train_apply_x.pkl")
    train_apply_y = joblib.load("features/train_apply_y.pkl")
    test_apply_x = joblib.load("features/test_apply_x.pkl")
    test_apply_y = joblib.load("features/test_apply_y.pkl")
    combined_apply_x_auto = joblib.load("features/combined_apply_x_auto.pkl")
    apply_y_auto = joblib.load("features/apply_y_auto.pkl")
    column_names_apply = joblib.load("features/column_names_apply.pkl")

    # train_analyze_x = joblib.load("features/train_analyze_x.pkl")
    # train_analyze_y = joblib.load("features/train_analyze_y.pkl")
    # test_analyze_x = joblib.load("features/test_analyze_x.pkl")
    # test_analyze_y = joblib.load("features/test_analyze_y.pkl")
    # combined_analyze_x_auto = joblib.load("features/combined_analyze_x_auto.pkl")
    # analyze_y_auto = joblib.load("features/analyze_y_auto.pkl")
    # column_names_analyze = joblib.load("features/column_names_analyze.pkl")

    # train_remember_x = joblib.load("features/train_remember_x.pkl")
    # train_remember_y = joblib.load("features/train_remember_y.pkl")
    # test_remember_x = joblib.load("features/test_remember_x.pkl")
    # test_remember_y = joblib.load("features/test_remember_y.pkl")
    # combined_remember_x_auto = joblib.load("features/combined_remember_x_auto.pkl")
    # remember_y_auto = joblib.load("features/remember_y_auto.pkl")
    # column_names_remember = joblib.load("features/column_names_remember.pkl")
    

    
if __name__ == "__main__":
    main()
    