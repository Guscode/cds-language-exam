# Assignment 4: Creating reusable network analysis pipeline

The goal of the assignment is to use the dataset real_and_fake_news.csv for creating a network of entities and their shared sentiment scores by sentence. create_edges.py will create edgelists and polarity scores, which can be used in networks.py to output network plots and network metrics.

### How to run

To run this code, please follow the guide for activating the virtual environment in [cds-language-exam](https://github.com/Guscode/cds-language-exam).

To test the script, in the virtual environment, please run:
```bash
cd assignment_4
python networks.py --path data/REAL_edges.csv --threshold 25 --metric weight
```
This will return a network plot in the folder viz, and a dataframe with degree, eigenvector and betweenness scores.

Figure 1:
<a href="https://github.com/Guscode/cds-language-exam">
    <img src="/assignment_4/viz/REAL_edges_network.png" alt="Logo" width="600" height="600">
</a>

Doing the same using fake news:

```bash
cd assignment_4
python networks.py --path data/FAKE_edges.csv --threshold 25 --metric weight
```

Figure 2:
<a href="https://github.com/Guscode/cds-language-exam">
    <img src="/assignment_4/viz/FAKE_edges_network.png" alt="Logo" width="600" height="600">
</a>

# User defined arguments

The user defined arguments are:

```bash
--path #Path to an edgelist in .csv format
--output # threshold for how many times a node-pair has to be mentioned to be included.
--metric # create network based on either weight or polarity scores
```

# Creating the edgelists
In order to create the edgelists, the script create_edges was made. Create edges takes a dataframe with a text column as input, and outputs by sentence entity-pairs with polarity scores per sentence. The script also allows the user to subset the data by a specific label in a 'label' column, which enables splitting on e.g. real and fake news.

Example usage:
```bash
python create_edges.py --data data/fake_or_real_news_small.csv --type REAL --output data/
```

# Methods

To extend this assignment, a command-line tool was created, that can create a network of entities in terms of co-occurrence on sentence-level, which also shows the mean sentiment of the sentences in which they co-occur. To build this tool, two scripts were created. Firstly, the script create_edges.py was created, which takes the dataset used in class with fake and real news articles and creates edges for either fake news or real news. Subsequently, using the Vader sentiment analysis tool (Hutto & Gilbert, 2014), the polarity score of each sentence is stored with the entity-pairs. The first script outputs a dataframe in .csv format with all entity-pairs and the accompanying polarity score. The second script, networks.py, takes this dataframe as input and creates a network plot. It does this by counting how many times each entity-pair occurs together, and subsequently creates a mean polarity-score per entity-pair. This is then plotted using matplotlib’s network functions, with green edges for entity-pairs with a positive mean polarity-score, black edges for entity-pairs with a neutral mean polarity score, and red edges for entity-pairs with a negative mean polarity score. Similarly, the script outputs a dataframe with the nodes and their network metrics. The network metrics include degree centrality, eigenvector centrality and betweenness centrality.


# Discussion

As seen in figure 1 and 2, the networks show mostly relations between American politicians. Contrary to research on sentiment in fake and real news (Zaeem et al., 2020), the real news shows more negative edges than the fake news. As the red edges display a negative co-occurrence pattern of the connected nodes, the fake news network would be hypothesized to show more red edges. In the figure 2, all edges are positive except between the nodes ‘Hillary Clinton’ and ‘Obama’, which is neutral. As a majority of fake news sources are pro-republican (Osmundsen et al., 2020), it makes good sense that this edge is not positive. A reason for the lack of negative edges can possibly be explained by what Dodds et al. (2015) has dubbed a universal positivity bias in human language. However, this theory is based on positivity arising as a means of peaceful and successful communication in conversation, while, generally, news articles show a negativity bias (Soroka et al., 2015). This complicates the interpretation of mean sentiment scores over many sentences, as both positivity bias and negativity bias play a role, but it seems that more occurrences is indicative of a high mean polarity score. 

When it comes to network metrics, table 1 shows how Hillary Clinton has both the highest eigenvector centrality score, which indicates a high degree of influence in the network, and a tendency to be connected to other important nodes. This is, however also driven by the fact that the entity-extractor in the create_edges.py script interprets ‘Clinton’, ‘Hillary Clinton’ and ‘Hillary’ as three separate persons, despite all referring to the same person. However, a difference is still observed between real and fake news, as real news tends to include both Hillary Clinton and Donald Trump, whereas Donald Trump’s eigenvector centrality score in fake news is below Hillary Clinton’s vice chair Huma Abedin. Betweenness Centrality describes a nodes communicative position in the network, indicating how nodes are not necessarily influential in themselves, but can play a facilitating 
role. Here, the same pattern is observed in terms of Hillary fixation in fake news and a broader range of nodes in real news.

Table 1:
<img width="417" alt="Screenshot 2021-05-26 at 14 07 44" src="https://user-images.githubusercontent.com/35924673/119657079-c309c280-be2b-11eb-903f-9b17096a711d.png">




