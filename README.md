<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/Guscode/cds-visual-exam">
    <img src="Cultural_language.jpeg" alt="Logo" width="247" height="154">
  </a>
  
  <h1 align="center">Cultural Data Science 2021</h1> 
  <h3 align="center">Language Analytics Exam</h3> 


  <p align="center">
    Gustav Aarup Lauridsen 
    <br />
  <p align="center">
    ID: au593405 
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
      <li><a href="#assignment-2---Collocates">Assignment 2 - Collocates</a></li>
      <li><a href="#assignment-4---Networks">Assignment 4 - Networks</a></li>
      <li><a href="#assignment-5---Hatespeech">Assignment 5 - Hatespeech</a></li>
      <li><a href="#assignment-6---Game_of_Thrones_Classification">Assignment 6 - Game of Thrones Classification</a></li>
      <li><a href="#self-assigned-project">self-assigned project</a></li>
    </li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- PROJECT INFO -->
## Project info

This repository contains assignments and descriptions regarding an exam in cultural data science at Aarhus University [_language Analytics_](https://kursuskatalog.au.dk/en/course/101990/Language-Analytics). The four class assignments are included in this repository, and the self-assigned project is found [here](https://github.com/Guscode/DKbert-hatespeech-detection).

The class assignments included in this portfolio are:
* Assignment 2 - _Collocates_
* Assignment 4 - _Networks_
* Assignment 5 - _Self-assigned hatespeech_
* Assignment 6 - _Game of Thrones Classification_

<!-- HOW TO RUN -->
## How to run

All scripts have been created and tested using python 3.8.6 on Linux
To run the assignments, you need to go through the following steps in your bash-terminal to configure a virtual environment on Worker02 (or your local machine) with the needed prerequisites for the class assignments:

__Setting up virtual environment and downloading data__
```bash
cd directory/where/you/want/the/assignment
git clone https://github.com/Guscode/cds-language-exam.git
cd cds-language-exam
bash create_lang_venv.sh
source langvenv/bin/activate
```

### Assignment 2 - Collocates

Go through the following steps to run assignment 2:

This code will extract collocates from a dataset with wine reviews.

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


For details and results see [```assignment_2/README.md```](https://github.com/Guscode/cds-language-exam/tree/main/assignment_2)

### Assignment 4 - Networks

Go through the following steps to run assignment 4:

The goal of the assignment is to use the dataset real_and_fake_news.csv for creating a network of entities and their shared sentiment scores by sentence. create_edges.py will create edgelists and polarity scores, which can be used in networks.py to output network plots and network metrics.

```bash
cd Assignment_4
python networks.py --path data/REAL_edges.csv --threshold 25 --metric weight
```

Doing the same for fake news:
```bash
cd Assignment_4
python networks.py --path data/FAKE_edges.csv --threshold 25 --metric weight
```

This will return a network plot in the folder viz, and a dataframe with degree, eigenvector and betweenness scores.


For details and results see [```assignment_4/README.md```](https://github.com/Guscode/cds-language-exam/tree/main/assignment_4)

### Assignment 5 - Hatespeech

Made in collaboration with [Johan Horsmans](https://github.com/JohanHorsmans)

Go through the following steps to run assignment 5:

This code will train and validate a hatespeech classification model using an ensemble model

```bash
cd assignment_5
python3 HateClass.py
```

For details and results see [```assignment_5/README.md```](https://github.com/Guscode/cds-language-exam/tree/main/assignment_5)


### Assignment 6 - Game of Thrones Classification

This code will train and validate a multinomial logistic regression and a convolutional neural network to predict which season of Game of Thrones a specific line comes from.

```bash
cd Assignment_6
python GOT_classification.py --input-path data/Game_of_Thrones_Script.csv --output output --epochs 10 --names include
```

With extra regularization to prevent overfitting, run:

```bash
python GOT_classification.py --input-path data/Game_of_Thrones_Script.csv --output output --epochs 10 --names include --model cnn --dropout True --regularizer 1e-7
```

For details and results see [```assignment_6/README.md```](https://github.com/Guscode/cds-language-exam/tree/main/assignment_6)

### self-assigned project

Made in collaboration with [Johan Horsmans](https://github.com/JohanHorsmans).

The self-assigned project is hosted in this repo [```self_assigned_project```](https://github.com/Guscode/DKbert-hatespeech-detection)

### Acknowledgements

* [Ross Deans Kristensen-McLachlan](https://github.com/rdkm89) for teaching and invaluable coding help
* [Frida Hæstrup](https://github.com/frillecode) for help and support and some wine 
* [Marie Mortensen](https://github.com/marmor97) for big brain energy 
* [Johan Horsmans](https://github.com/JohanHorsmans) for collaborating on assignment 5 and the self-assigned project, as well as helping with readme structure and being a stand up guy
* [Emil Jessen](https://github.com/emiltj) for helping with readme structure and being a stand up guy

## Contact
For contact, please reach out to me on:  201804481@post.au.dk





