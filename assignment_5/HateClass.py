#!/usr/bin/env python

# System tools:
import os
import sys
sys.path.append(os.path.join(".."))

# Data munging tools:
import pandas as pd
import numpy as np
import utils.classifier_utils as clf
from utils.neuralnetwork import NeuralNetwork


# Machine learning stuff:
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics
from sklearn.metrics import balanced_accuracy_score, classification_report, plot_confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree, svm
from sklearn.neural_network import MLPClassifier

# Set seed for reproducibility:
import random
random.seed(10)

# Define the main function of the script: 
def main():
    
    print("[INFO]: Loading and preprocessing data")
    
    # Specify filepaths to load training- and testing data:
    filepath = os.path.join("data","Train_Hate.tsv")
    filepath_test = os.path.join("data", "Test_Hate.tsv")

    # Load Data:
    Data = pd.read_csv(filepath, index_col=0, sep="\t")
    Data_test = pd.read_csv(filepath_test, index_col=0, sep="\t")
    
    # Preprocessing: Dropping NA's:
    Data = Data.dropna()
    Data_test = Data_test.dropna()

    # Converting label-column from "NOT" an "OFF" to 0's and 1's, respectively: 
    Data["subtask_a"] = np.where(Data["subtask_a"] == 'NOT', 0, 1)
    Data_test["subtask_a"] = np.where(Data_test["subtask_a"] == 'NOT', 0, 1)
    
    # Dividing data into X and y variables:
    X_train = Data["tweet"]
    y_train = Data["subtask_a"]
    X_test = Data_test["tweet"]
    y_test = Data_test["subtask_a"]
    
    # Vectorizing to make lowercased unigrams and bigrams with tf-idf transformation and remove 
    #-frequently used words:
    vectorizer = TfidfVectorizer(ngram_range = (1,2),
                             lowercase =  True,
                             max_df = 0.95)

    # Vectorize training data:
    X_train_feats = vectorizer.fit_transform(X_train)
    # Vectorize testing data:
    X_test_feats = vectorizer.transform(X_test)
    # List feature names:
    feature_names = vectorizer.get_feature_names()
    
    
    print("[INFO]: Building-, training- and testing models")
    
    # Build classifiers:
    
    # LOGISTIC REGRESSION:
    
    # Fit Logistic Regression model:
    lr_classifier = LogisticRegression(random_state=42).fit(X_train_feats, y_train)

    # Get Logistic Regression predictions:
    lr_predictions = lr_classifier.predict(X_test_feats)

    # RANDOM FOREST:
    
    # Instantiate random forest model with 1000 decision trees:
    rf = RandomForestRegressor(n_estimators = 100, random_state = 42)
    
    # Train the model on training data:
    rf.fit(X_train_feats, y_train);

    # Get Random Forest predictions:
    rf_predictions = rf.predict(X_test_feats)
    
    # SVM:

    # Initialize SVM classifier:
    clf_svm = svm.SVC(kernel='linear')
    
    # Fit SVM:
    clf_svm = clf_svm.fit(X_train_feats, y_train)
    
    # Get SVM predictions:
    svm_predictions = clf_svm.predict(X_test_feats)


    # DECISION TREE:
    
    # Initialize model:
    clf_dt = tree.DecisionTreeClassifier()
    
    # Fit Decision tree:
    clf_dt = clf_dt.fit(X_train_feats, y_train)

    # Get Decision tree predictions
    dt_predictions = clf_dt.predict(X_test_feats)

    # NEURAL NETWORK:

    # Initialize Neural Network:
    clf_nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                         hidden_layer_sizes=(256, 128), random_state=1)


    # Fitting Neural network:
    clf_nn = clf_nn.fit(X_train_feats, y_train)

    # Get neural network predictions:
    nn_predictions = clf_nn.predict(X_test_feats)

    print("[INFO]: Ensembling")
    
    # Creating ensemble predictions by adding all model-predictions together and setting a threshold of 1.4 (i.e. if accumulated probability of a comment being offensive exceeds 1.4, classify it as offensive. Otherwise, classify it as non-offensive:
    preds = np.where(rf_predictions+nn_predictions+dt_predictions+lr_predictions > 1.4, 1, 0)

    print("[INFO]: Confusion Matrix:")
    print(pd.crosstab(y_test, preds))

    print("[INFO]: Classification report:")
    print(classification_report(y_test, preds))
    
    print("A macro F1-score of .71 would place us in a 23rd place (out of 38) in the OffensEval2020-competition. Not bad for a computationally inexpensive ensemble model.")
    
if __name__=="__main__":
    main()