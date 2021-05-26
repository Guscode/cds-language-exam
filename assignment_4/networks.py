#!/usr/bin/env python

"""
Produce network graph and metrics df from edgelist dataframe including columns 'nodeA' and 'nodeB'.
Parameters:
    path: edgelist.csv
    threshold: 15

Example:
    $ python networks.py -p edgelist.csv -t 15

Find virtual environmnet at github.com/guscode/networks_assignment
"""

# Load packages
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import argparse
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from collections import Counter

import networkx as nx
import scipy
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (3,3)

class create_network:
    def __init__(self,args):
        self.args = args
        self.data = pd.read_csv(self.args["path"])  
        
    #Function for reading an edgelist and turning it into a weighted dataframe with a threshold
    def read_edgelist(self):
        #defining which columns are needed
        edges_df = self.data[["nodeA", "nodeB"]]
        edges_df = edges_df[edges_df["nodeA"] != edges_df["nodeB"]]
        #converting dataframe into list of tuples
        edge_tuples = [tuple(x) for x in edges_df.values]

        #Creating empty list for edge counts
        counted_edges = []
        
        if "polarity" not in self.data:
            self.data["polarity"] = [0]*len(self.data["nodeA"])
        
        #looping through counted list of tuples, saving nodes and weight in counted_edges list
        for pair,weight in Counter(edge_tuples).items():
            nodeA = pair[0]
            nodeB = pair[1]
            sub_df = self.data[self.data["nodeA"]==nodeA]
            sub_df = sub_df[sub_df["nodeB"] == nodeB]
            counted_edges.append((nodeA,nodeB,weight,sub_df["polarity"].mean()))
            

        #Converting counted_edges to dataframe
        edges_df = pd.DataFrame(counted_edges, columns=["nodeA", "nodeB", "weight", "polarity"])

        #Filtering weight by threshold
        self.filtered_df = edges_df[edges_df["weight"]>self.args["threshold"]]
        
    def plot_network(self):
        
        #creating list with color names based on polarity of edges
        colors = np.where(self.filtered_df["polarity"] > 0.01, "green", "black")
        colors = np.where(self.filtered_df["polarity"] < -0.01, "red", colors)
        
        #Creating a network with networkx and edgelist_df
        self.G = nx.from_pandas_edgelist(self.filtered_df, "nodeA", "nodeB", [self.args["metric"]])
        
        #plotting network
        pos = nx.drawing.nx_pylab.draw_spring(self.G,node_size=5, with_labels=True, font_size=4, edge_color = colors)
        fig1 = plt.gcf()

        #saving network graph
        outpath = os.path.join("viz", Path(self.args["path"]).stem+"_network.png")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        print(f"saved network graph at {outpath}")

    #Function for extracting network centrality metrics from network object
    def save_metrics_df(self):
        #creating list of all metrics
        all_metrics = [nx.degree_centrality(self.G), nx.eigenvector_centrality(self.G),nx.betweenness_centrality(self.G)]
        #creating empty dict for saving metrics
        all_metrics_dict = {}
        #formatting metrics into tuples
        for k in nx.degree_centrality(self.G).keys():
            all_metrics_dict[k] = tuple(m[k] for m in all_metrics)

        #Making dataframe
        metrics_df = pd.DataFrame.from_dict(all_metrics_dict, orient="index", columns = ["degrees", "eigenvector", "betweenness"])

        #Making sure the index is numbers, and creating a column called 'node' with all the nodes.
        metrics_df["node"]=metrics_df.index
        metrics_df.index = [i for i in range(len(metrics_df))]
        outpath = os.path.join("viz", Path(self.args["path"]).stem+"_metrics.csv")
        metrics_df.to_csv(outpath)

def main():

    #Create ouput folder if it isn't there already
    if not os.path.exists("viz"):
        os.makedirs("viz")
        
    # Define function arguments 
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required = True, help= "Path to edgelist.csv")
    ap.add_argument("--threshold", required = False,default = 10, type = int, help= "threshold for weights")
    ap.add_argument("--metric", required = False, default = "weight", help= "metric for network, options = weight, polarity")

    args = vars(ap.parse_args())

    #Execute read function - returns dataframe with weighted edges
    network = create_network(args)
    network.read_edgelist() 
    network.plot_network()
    network.save_metrics_df()

# Define behaviour when called from command line
if __name__=="__main__":
    main()
