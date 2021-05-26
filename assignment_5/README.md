<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/JohanHorsmans/cds-language-exam-2021">
    <img src="../README_images/nlp2.png" alt="Logo" width="150" height="150">
  </a>
  
  <h1 align="center">Cultural Data Science 2021</h1> 
  <h3 align="center">Assignment 5</h3> 

  <p align="center">
    Johan Kresten Horsmans
    <br />
    <a href="https://github.com/JohanHorsmans/cds-language-exam-2021/blob/main/Language_Analytics_Exam.pdf"><strong>Link to PDF with all portfolio descriptions »</strong></a>
    <br />
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#official-description-from-instructor">Official description from instructor</a></li>
    <li><a href="#contribution">Contribution</a></li>
    <li><a href="#methods">Methods</a></li>
    <li><a href="#how-to-run">How to run</a></li>
    <li><a href="#repository-structure-and-contents">Repository structure and contents</a></li>
    <li><a href="#discussion-of-results">Discussion of results</a></li>
  </ol>
</details>

<!-- OFFICIAL DESCRIPTION FROM INSTRUCTOR -->
## Official description from instructor

### Applying (un)supervised machine learning to text data

The assignment this week involves a little bit of a change of pace and a slightly different format. As we have the Easter break next week, you also have a longer period assigned to complete the work.

For this task, you will pick your own dataset to study.

This dataset might be something to do with COVID-19 discourse on Reddit; IMDB reviews; newspaper headlines; whatever it is that catches your eye. However, I strongly recommend using a text dataset from somewhere like Kaggle - https://www.kaggle.com/datasets

When you've chosen the data, do one of the following tasks. One of them is a supervised learning task; the other is unsupervised. 

__NOTE: I HAVE CHOSEN THE FOLLOWING TASK:__

* Train a text classifier on your data to predict some label found in the metadata. For example, maybe you want to use this data to see if you can predict sentiment label based on text content.

You should formulate a short research statement explaining why you have chosen this dataset and what you hope to investigate. This only needs to be a paragraph or two long and should be included as a README file along with the code. E.g.: I chose this dataset because I am interested in... I wanted to see if it was possible to predict X for this corpus.

In this case, your peer reviewer will not just be looking to the quality of your code. Instead, they'll also consider the whole project including choice of data, methods, and output. Think about how you want your output to look. Should there be visualizations? CSVs?

You should also include a couple of paragraphs in the README on the results, so that a reader can make sense of it all. E.g.: I wanted to study if it was possible to predict X. The most successful model I trained had a weighted accuracy of 0.6, implying that it is not possible to predict X from the text content alone. And so on.

__Tips__
* Think carefully about the kind of preprocessing steps your text data may require - and document these decisions!
* Your choice of data will (or should) dictate the task you choose - that is to say, some data are clearly more suited to supervised than unsupervised learning and vice versa. Make sure you use an appropriate method for the data and for the question you want to answer
* Your peer reviewer needs to see how you came to your results - they don't strictly speaking need lots of fancy command line arguments set up using argparse(). You should still try to have well-structured code, of course, but you can focus less on having a fully-featured command line tool

__General instructions__
* You should upload standalone .py script(s) which can be executed from the command line
* You must include a requirements.txt file and a bash script to set up a virtual environment for the project You can use those on worker02 as a template
* You can either upload the scripts here or push to GitHub and include a link - or both!
* Your code should be clearly documented in a way that allows others to easily follow the structure of your script and to use them from the command line

__Purpose__
This assignment is designed to test that you have an understanding of:

* how to formulate research projects with computational elements;
* how to perform (un)supervised machine learning on text data;
* how to present results in an accessible manner.

<!-- CONTRIBUTION -->
## Contribution
I have carried out the following assignment with [Gustav Aarup Lauridsen](https://github.com/Guscode). Gustav and I contributed equally to every stage of this project from initial conception and implementation, through the production of the final output and structuring of the repository. (50/50%)

<!-- METHODS -->
## Methods
We have chosen the OffensEval2020 dataset containing 3000+ Danish comments from Ekstra Bladet and Reddit, labeled with a binary coding scheme indicating offensiveness (link: https://figshare.com/articles/dataset/Danish_Hate_Speech_Abusive_Language_data/12220805).

OffensEval2020 was a competition where researchers and data scientists from all over the world competed to create the best classification models for various languages (including Danish).

The best team in the Danish task achieved a macro F1-score of 0.8119 and the worst team achieved a score of 0.4913. For the full paper, see: https://arxiv.org/pdf/2006.07235.pdf

We wanted to create a text classifier that could classify offensive comments in Danish and compare our macro F1-score with the results from the OffensEval2020 competition. We found this task interesting due to the amount of media coverage on Danish hate speech on social media in recent months. We believe that a robust hate-speech classifier could be a very valuable tool for moderating the tone and retoric of the public debate to make it more constructive.   

We trained the following models: Logistic Regression, Support Vector Machine, Neural Network, Random Forest & Decision Tree (see code for further specifications). We then combined them in an ensemble where a comment was classified as offensive if the accumulated probability of all the model classifications exceeded 1.4. We have chosen to use the macro F1-score as our evaluation metric:

The F1-score is a metric devised to fuse the relation between model precision and recall into a unified score. The metric is defined as taking the harmonic mean of precision and recall. The reason for using the harmonic mean, rather than the arithmetic mean, is that the harmonic mean of a recall-score of 0 and a precision-score of 100 would result in an F1-score of 0, rather than 50. This is advantageous, since it means that a model cannot achieve a high F1-score by having a high recall or precision by itself. The macro-averaging procedure of the macro F1-score involves calculating the arithmetic mean of the F1-score for each class. 

To ensure reproducible results, we used random.seed.

<!-- HOW TO RUN -->
## How to run

__NOTICE:__ To run the assignment, you need to have configured and activated your virtual environment. See the main [README](https://github.com/JohanHorsmans/cds-language-exam-2021/blob/main/README.md) for a guide on how to do this.

Go through the following steps to run assignment 5:
```bash
cd {root directory (i.e. cds-language-exam-2021)}
cd assignment_5
python3 HateClass.py
```
<!-- REPOSITORY STRUCTURE AND CONTENTS -->
## Repository structure and contents

This repository contains the following folder:

|Folder|Description|
|:--------|:-----------|
```data/``` | Folder containing a testing- and training dataset consisting over a 3.000 social media comments labeled after offensiveness (i.e. _NOT_ and _OFF_).

Furthermore, it holds the following files:
|File|Description|
|:--------|:-----------|
```HateClass.py``` | The python script for the assignment.
```README.md``` | The README file that you are currently reading.

<!-- DISCUSSION OF RESULTS -->
## Discussion of results

Our ensemble containing all models achieved a macro F1-score of 0.71. It is important to note that the dataset is heavily skewed towards non-offensive comments. The skewed data is also reflected in our model predictions, where the F1-score was much higher for non-offensive comments compared to offensive ones (0.95 vs. 0.46). We believe that this bias towards non-offensive comments, might very well reflect the imbalanced nature of the dataset. It defeats the purpose of a hate-speech classifier if it categorizes such a large proportion of hate as non-offensive and, as such, we argue that there is still a lot of work to be done in spite of the seemingly honorable F1-score of 0.71.

Nonetheless it would have ranked as the 23rd best model (out of 38) in the OffensEval2020 competition, so we deem it to be quite successful when taking the circumstances into account. For an even better model, see our [self-assigned project](https://github.com/JohanHorsmans/cds-language-exam-2021/tree/main/self_assigned), where we achieve a macro F1-score of 0.78 on the same dataset.

<br />
<p align="center">
  <a href="https://github.com/JohanHorsmans/cds-visual-exam-2021">
    <img src="../README_images/logo_au.png" alt="Logo" width="300" height="102">
  </a>
