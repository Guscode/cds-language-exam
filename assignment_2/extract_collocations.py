'''
Use the function get_collocations to extract how many times
words occur together in a corpus, both in raw frequency, and
In terms of a mutual information score. 

The function takes the following arguments:
data: path to your dataframe in csv format
column: specify which column includes text, default = text
Word: the target word, for which you wish to find collocations
Window: Defines how closely the target word and the collocation
        has to interact. e.g. a window=5 will search for collocations
        5 words before and after the target word.

The function returns a pandas dataframe with the columns:
raw_freq: How many times the target word and collocation 
          occured together.
MI: Mutual information score based on calculations from
    http://www.collocations.de/AM/section1.html
And prints the ten words with the highest MI score.
            
'''

#Importing packages
import re
import argparse
import string
import pandas as pd
import math
import os


def get_collocations(texts, word = "", window = 5): 
    all_collocations = pd.DataFrame(columns=["raw_freq", "MI", "Col"]) #Making empty dataframe to store results

    clean_text = [re.sub(r"\W+", " ", txt) for txt in texts] #Cleaning the texts so they only include letters and spaces
    
    all_words=[] #Creating empty list for all words present in the corpus

    for txt in clean_text:
        all_words= all_words+txt.split() #Loop through each text and extract all the words
        
    all_words = list(set(all_words)) #Select unique words by making the list into a set, and back
    all_words.remove(word) #Remove the target word from the list
    
    for col in all_words: #Loop through all words in the corpus
        res_dict = {'wordcol':0, 'no_wordcol':0, 'no_wordno_col':0, 'wordno_col':0} #Make empty dictionary to store results
                                                                                    
        MI=0 #Set MI score to 0
        
        for text in texts: #Loop through each text in the corpus 
            word_indicator = "no_word" #Create indicator of whether the target word is present
            collocate_indicator ="no_col" #Create indicator of whether the collocate is present
            split_text = text.split() #Split the text
            if word in split_text: #If the target word is in the text do the following
                    word_indicator = "word" #Change target word_indicator from 'no_word' to 'word'
                    start = max(0,split_text.index(word)-window) #Make start variable to indicate the start of the window
                    end = split_text.index(word)+window #Make end variable to indicate the end of the window
                    shortened_text = split_text[start:end] #Make the window
                    if col in shortened_text: #If the collocate is in the window
                        collocate_indicator = "col" #Change collocate_indicator from 'no_col' to 'col'
                        
            if text.find(col)!= -1 and word_indicator=="no_word": #If collocate is in the text and target word isn't do this 
                collocate_indicator = "col" #Change collocate_indicator from 'no_col' to 'col'

            res_dict[word_indicator+collocate_indicator] += 1 #Change value in res_dict (all texts will have a word_indicator
                                                              #of either 'word' or 'no_word' and a collocate_indicator of either
                                                              #'col' or 'no_col'. When they are pasted together, they represent 
                                                              #the four categories in the res_dict 
        
        R1 = res_dict["wordcol"]+res_dict["wordno_col"] #Calculate R1-score
        C1 = res_dict["no_wordcol"]+res_dict["wordcol"] #Calculate C1-score
        C2 = res_dict["wordno_col"]+res_dict["no_wordno_col"] #Calculate C2-score
        E11 = (R1*C1)/(C1+C2) #Calculate E11-score
    
        if E11 > 0: #if statements to avoid dividing by zero
            if res_dict["wordcol"]>0:
                MI = math.log((res_dict["wordcol"]/E11)) #Calculate MI-score

        raw_freq = res_dict["wordcol"] #Extract raw frequency (when word and collocate appear together)
        all_collocations_length = len(all_collocations) #Add info to dataframe
        all_collocations.loc[all_collocations_length] = [raw_freq, MI,col]
        
    
    return all_collocations #return dataframe

def main():
    
    if not os.path.exists("out"):
        os.makedirs("out")
        
     #Add the terminal argument
    ap = argparse.ArgumentParser()

    #Let users define # of epochs
    ap.add_argument("--data", required = True, type = str,
                    help="Specify path to dataframe i csv format" )
    ap.add_argument("--column", required=False,type = str, default = "text",
                    help="column-name including text")
    ap.add_argument("--word", required=True,type = str,
                    help="primary word for collocate search")
    ap.add_argument("--window", required=False,type = int, default=5,
                    help="primary word for collocate search")
    
    #parse arguments
    args = vars(ap.parse_args())
    
    #read data
    data = pd.read_csv(args["data"])
    
    #select column with text
    texts = data[args["column"]]
    #Define target word
    target_word = args["word"]
    #Use function get collocates, which returns all collocations in a dataframe
    collocations = get_collocations(texts, word = target_word, window = args["window"])
    #Sort dataframe by MI-score
    collocations = collocations.sort_values("MI", ascending=False)
    #print top five words
    print(f"Top five collocates for: {target_word}")
    print(collocations[["MI", "Col"]][:10])
    collocations.to_csv("out/all_collocations.csv")
    
if __name__=="__main__":
    main()