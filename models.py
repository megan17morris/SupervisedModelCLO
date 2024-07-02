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
    original = pd.read_excel("data/cleaned_original_LWIC.xlsx")
    auto = pd.read_excel("data/cleaned_auto_LWIC.xlsx")
    metric = load_metric("f1")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="macro")
    
if __name__ == "__main__":
    main()