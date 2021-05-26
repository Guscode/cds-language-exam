# Assignment 2: String processing with Python

Using a text corpus found on the cds-language GitHub repo or a corpus of your own found on a site such as Kaggle, write a Python script which calculates collocates for a specific keyword.

The script extract_collocations.py will take a dataframe with a text column and a target word as input, and output a dataframe with collocates. In order to test the script, a dataset of wine descriptions was found on [kaggle](https://www.kaggle.com/zynicide/wine-reviews). Using the script on this data will help you to sound like a connoisseur at any wine tasting.

### How to run

To run this code, please follow the guide for activating the virtual environment in [cds-language-exam](https://github.com/Guscode/cds-language-exam).

To test the script, in the virtual environment, please run:
```bash
cd Assignment_2
python extract_collocations.py --data data/10kWines.csv --column description --word fruity --window 5
```
This will return a dataframe called all_collocations in the folder 'out'. Similarly, the script will print the ten strongest collocations based on mutual information score. The top five for target word fruity is:

- breathtakingly
- Salmon 
- Done 
- Exotically 
- Whimsically 

# User defined arguments

The user defined arguments are:

```bash
--data #Path to a dataset in .csv format
--column #Name of column with text
--word # Target word
--window # amount of words before and after target word is taken as a collocate.
```

# Methods

In order to solve this challenge, a script was made which takes four inputs from terminal arguments using argparse. Firstly, the user specifies the path to a data frame in .csv format, and a column argument stating which column contains the text the user wants to extract collocates from. Similarly, the script takes the target word as input from the terminal and the window size of the collocate finder. The window size determines how close words have to appear together in order to make them collocates. In order to test the tool, a dataset with wine descriptions from Kaggle.com was used. Using this dataset, the user is able to search for specific taste characteristics and extract which tastes, or notes are related to the specific characteristic. The script firstly extracts all unique words from the dataset and searches in each text by the specified window size, if the target word and another word appear together or not. Using these counts, a mutual information score is calculated for each collocate pair, which creates a sophisticated measure of co-occurrence, as it also takes non-occurrences into account. In contrast to using raw frequency of co-occurrence, the MI-score is not relying on the words being used often, rather it is a measure of how much they are used together versus separately. The script prints the ten strongest collocations and save alle collocations in a data frame in .csv format.

# Discussion

Running the script on the wine data with the target word “fruity” and a window size of 5, returned ten words: 

’breathtakingly’, ‘Salmon’, ‘Done’, ‘Exotically’, ‘Whimsically’, ‘quintas’, ‘Errazuriz’, ‘appealingly’, ‘Cognac’, ’Vacheron’.

These words include great adjectives and adverbs that can be used to describe a fruity taste, as well as regions of wine production and suggestions for dishes that fit well with fruity wine. Aside from these, the collocate ‘Done’ is also included, which is not immediately related to the word fruity. Why the word ‘Done’ is included can be examined by including n-grams in the collocate search or diving deeper into the text. However, with nine out of ten words making good sense and allowing the user to sound like a wine expert, the script can be a very useful tool.





