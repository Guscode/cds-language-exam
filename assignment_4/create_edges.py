'''
Use the function create_edges to turn the fake_or_real_news dataset
into edges which can be visualized in a network. This function
creates a dataframe with entity pairs occuring together in a sentence
and the polarity score of each sentence.

The function takes the following arguments:
data: path to fake_or_real_news.csv
type: Type of news to create edges, can be REAL or FAKE
output: Path to output folder.
'''

# System tools
import os
import argparse

# Data analysis
import pandas as pd
from collections import Counter
from itertools import combinations 
from tqdm import tqdm

# NLP
import spacy
nlp = spacy.load("en_core_web_sm")
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

#Define main function
def main():
    
    #Add terminal arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required = True, help= "Path to fake_or_real dataset")
    ap.add_argument("--label-spec", required = True, help= "specify if data should be split by spcific label")
    ap.add_argument("--type", required = False, default = "REAL", help= "type of news, default = True, options, REAL, FAKE")
    ap.add_argument("--output", required = False, default = "..", help= "define output path")

    args = vars(ap.parse_args())
    
    #Read data
    data = pd.read_csv(args["data"])
    
    #Filter article type
    if args["label_spec"] is not None:
        real_df = data[data["label"]==args["type"]]["text"]
    else:
        real_df = data["text"]
    
    
    #Create empty lists for storing entities and polarity
    text_entities = []
    polarity = []

    #Loop through all texts in dataset
    for text in tqdm(real_df):
        doc = nlp(text) #Create sentences
        for sentence in doc.sents: #loop through sentences in each article
            score = analyser.polarity_scores(sentence.text)["compound"] #extract polarity score
            tmp_entities = [] #Create empty list for emporary entity storage
            for entity in sentence.ents: #For all entities in each sentence  
                if entity.label_ == "PERSON":# if that entity is a person
                    tmp_entities.append(entity.text)# append to temp list
            if len(tmp_entities)>=2: #If there are more than two entities (a co-occurance)
                text_entities.append(tmp_entities) #add entities to list
                polarity.append(score) #add polarity score to list
    
    edgelist = [] #Create empty list for storing edges
    # iterate over every document
    for text, score in zip(text_entities, polarity):
        # use itertools.combinations() to create edgelist
        edges = list(combinations(text, 2))
        # for each combination - i.e. each pair of 'nodes'
        for edge in edges:
            # append this to final edgelist
            edgelist.append([tuple(sorted(edge)), score])
    
    #Create dataframe from edges and polarity score
    df = pd.DataFrame(edgelist, columns = ["edge", "polarity"])
    #Extend edges to two columns named nodeA and nodeB
    df[['nodeA', 'nodeB']] = pd.DataFrame(df['edge'].tolist(), index=df.index)
    df = df[['nodeA', 'nodeB', 'polarity']] #Choose only these three columns
    output_path = os.path.join(args["output"], "_".join([args["type"], "edges.csv"])) #create path for output
    df.to_csv(output_path) #save the dataframe
    
if __name__=="__main__": #Run if called from terminal
    main()
        
