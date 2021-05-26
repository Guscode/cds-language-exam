#! usr/bin/python

#Import libraries
import os
import sys
import argparse

sys.path.append(os.path.join(".."))
from utils import classifier_utils as clf

import pandas as pd
import numpy as np
import re 

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler,LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn import metrics

# tools from tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Embedding, 
                                     Flatten, GlobalMaxPool1D, Conv1D, Dropout)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.regularizers import L2



# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns


def plot_history(H, epochs):
    """
    Utility function for plotting model history using matplotlib
    
    H: model history 
    epochs: number of epochs for which the model was trained
    """
    plt.style.use("fivethirtyeight")
    plt.figure()
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    


def create_embedding_matrix(filepath, word_index, embedding_dim):
    """ 
    A helper function to read in saved GloVe embeddings and create an embedding matrix
    
    filepath: path to GloVe embedding
    word_index: indices from keras Tokenizer
    embedding_dim: dimensions of keras embedding layer
    """
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim)) #Create empty embedding matrix to fill in

    with open(filepath) as f: #open embedding file
        for line in f: #read line by line
            word, *vector = line.split() #Extract data
            if word in word_index: #Add words and embeddings to matrix
                idx = word_index[word] 
                embedding_matrix[idx] = np.array(
                    vector, dtype=np.float32)[:embedding_dim]

    return embedding_matrix


def save_heatmap(y_test, y_pred, outpath):
    """ 
    Function that creates and saves a classification matrix heatmap.
    """

    data = metrics.confusion_matrix(y_test, y_pred) #Create confusion matrix 

    data = data / data.sum(axis=1) #Convert classification counts into percentages
    data = np.round(data, decimals=3) #round percentages

    df_cm = pd.DataFrame(data, columns=np.unique(y_test), \
                         index = np.unique(y_test)) #Convert confusion matrix to dataframe
    df_cm.index.name = 'Actual' #Name index 
    df_cm.columns.name = 'Predicted' #name Columns
    plt.figure(figsize = (10,7)) #Set figure size 
    sns.set(font_scale=1.4) #Set label size
    hm = sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16}) #Create heatmap
    hm.figure.savefig(outpath) #save heatmap
    return True
    
class got_classification:
    """ 
    Class for classifying game of thrones seasons
    with a logistic regression and a convolutional 
    neural network
    """
    def __init__(self,args): #Init function
        self.args = args #Define command line arguments
        self.data = pd.read_csv(self.args["input_path"]) #Load data
        
    def preprocess(self): #Perform preprocessing
        self.data["Season"] = [re.sub(" ", "_", i) for i in self.data["Season"]]
        
        #including name of speaker in text if user specifies it
        if self.args["names"] == "include":
            self.data["text"] = self.data["Name"]+ " " +self.data["Sentence"]
            self.texts = self.data["text"].values
        else:
            self.texts = self.data["Sentence"]
        
        self.labels = self.data["Season"].values #Defining labels

        # Split into test/train
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.texts,#texts
                                                            self.labels,          #labels
                                                            test_size=self.args["test_split"],   #Split train/test 80/20
                                                            random_state=42)
        self.X_train = self.X_train.astype(str) #Convert to type string
        self.X_test = self.X_test.astype(str) #Convert to type string

    def perform_lr(self): #Perform logistic regression
        
        vectorizer = CountVectorizer()#add count-vectorizer

        self.X_train_feats = vectorizer.fit_transform(self.X_train) #Vectorize training data 
        self.X_test_feats = vectorizer.transform(self.X_test) #Vectorize testing data 
        
        pipe = make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=10000, random_state=42)) #create pipeline

        pipe.fit(self.X_train_feats, self.y_train) #use pipeline to scale and perform logistic regression

        y_pred = pipe.predict(self.X_test_feats) # predict test data
        
        outpath = os.path.join(self.args["output"], "LR_heatmap.jpg") #Create path for output
        save_heatmap(self.y_test, y_pred, outpath) #Save heatmap confusion matrix
        
        #Create Classification report and save as .csv file
        results_df = pd.DataFrame(metrics.classification_report(self.y_test, y_pred, output_dict=True)).transpose() 
        results_df = results_df.round(3)
        output_path = os.path.join(self.args["output"], "results_df_lr.csv")
        results_df.to_csv(output_path)

        
    def perform_cnn(self): #Function that performs CNN
        
        lb = LabelBinarizer() #Deine label binarizer
        y_train_cnn = lb.fit_transform(self.y_train) #Binarize training labels
        y_test_cnn = lb.fit_transform(self.y_test) #binarize testing labels
        
        tokenizer = Tokenizer(num_words=None) #Define Tokenizer
        
        tokenizer.fit_on_texts(self.X_train)
        
        X_train_toks = tokenizer.texts_to_sequences(self.X_train)  #Tokenize training texts
        X_test_toks = tokenizer.texts_to_sequences(self.X_test)  #Tokenize testing texts
     
        vocab_size = len(tokenizer.word_index) + 1  #Creating vocabulary size and adding 1 because of reserved 0 index
        
        maxlen = max([len(x) for x in self.X_train]) #adding max length of sentences

        
        X_train_pad = pad_sequences(X_train_toks, 
                                    padding='post',
                                    maxlen=maxlen) #Padding training data
 
        X_test_pad = pad_sequences(X_test_toks, 
                                   padding='post', 
                                   maxlen=maxlen) #Padding testing data
        embedding_dim = 50 #Define embedding size

        embedding_matrix = create_embedding_matrix("embeddings/glove.6B.50d.txt",
                                                   tokenizer.word_index, 
                                                   embedding_dim) #Creating embedding matrix
        
        
        l2 = L2(self.args["regularizer"]) #Define regularizer, lower values will prevent overfitting

        # Intialize sequential model
        model = Sequential()

        # Add embedding layer
        model.add(Embedding(input_dim = vocab_size,
                            output_dim = embedding_dim,
                            input_length = maxlen))    

        # Add convolutional layer
        model.add(Conv1D(256, 1,
                         activation = 'relu',
                         kernel_regularizer = l2)) # L2 regularization 

        # Global max pooling
        model.add(GlobalMaxPool1D())
        
        if self.args["dropout"] == "True":
            model.add(Dropout(0.2))
        

        # Add dense layer
        model.add(Dense(128, activation = 'relu',
                        kernel_regularizer = l2))

        # Add dense layer with 8 nodes; one for each season 
        model.add(Dense(8,
                        activation = 'softmax')) # we use softmax because it is a categorical classification problem

        # Compile model
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])


        history = model.fit(X_train_pad, y_train_cnn,
                            epochs=self.args["epochs"],
                            verbose=False,
                            validation_data=(X_test_pad, y_test_cnn),
                            batch_size=10) #Fit model to the training data

        # evaluate 
        loss, accuracy = model.evaluate(X_train_pad, y_train_cnn, verbose=False)
        print("Cnn model Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test_pad, y_test_cnn, verbose=False)
        print("Cnn model Testing Accuracy:  {:.4f}".format(accuracy))

        # plot history and save the plot
        plot_history(history, epochs = self.args["epochs"])
        plt.savefig(os.path.join(self.args["output"],"history_plot_cnn.jpg"))
        
        #Predict test set
        y_preds = model.predict(X_test_pad)

        outpath = os.path.join(self.args["output"],"cnn_heatmap.jpg") #create output path for confusion matrix
        
        ytest = [np.where(i == 1)[0][0] for i in y_test_cnn] #convert test labels to list format
        ypred = [np.argmax(i) for i in y_preds] #convert prediction labels to list format
        
        save_heatmap(ytest, ypred, outpath) #Make confusion matrix heatmap
        
        results_df = pd.DataFrame(metrics.classification_report(ytest, ypred, output_dict=True)).transpose() 
        results_df = results_df.round(3)
        output_path = os.path.join(self.args["output"], "results_df_cnn.csv")
        results_df.to_csv(output_path)


def main():
    #Add all the terminal arguments
    ap = argparse.ArgumentParser()

    ap.add_argument("--input-path", required = True,
                    help="Path to the GOT dataset" )
    ap.add_argument("--output", required = False,
                    default = "",
                    help="Add output path to store results in a different folder")
    ap.add_argument("--test_split", required = False,
                    default = 0.2, type = float,
                    help="Add size of test data. default = 0.2")
    ap.add_argument("--regularizer",required = False, default=0.001,
                    type=float,
                    help="L2 regularizer")
    ap.add_argument("--epochs", required = False,
                    default = 10, type=int,
                    help="Amount of epochs. Default=10")
    ap.add_argument("--model", required = False,
                    default = "both", type=str,
                    help="specify which model to run. default: both, options: lr, cnn")
    ap.add_argument("--names", required = False,
                    default = "include", type=str,
                    help="Decide if speaker names should be included in text. default = include, options: include, exclude")
    ap.add_argument("--dropout", required = False,
                    default = "False", type=str,
                    help="Decide if dropout layer should be included in cnn. default = False, options: False, True")
    
    #parse arguments
    args = vars(ap.parse_args())

    #Run everything
    classify = got_classification(args)
    classify.preprocess()
    if args["model"] == "cnn":
        classify.perform_cnn()
    elif args["model"] == "lr":
        print("[INFO]: performing logistic regression classification")
        classify.perform_lr()
    else:
        print("[INFO]: performing logistic regression classification")
        classify.perform_lr()
        print("[INFO]: performing classification with Convolutional Neural Network")
        classify.perform_cnn()

    
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()
    print("[INFO]: DONE!")