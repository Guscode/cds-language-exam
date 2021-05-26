# Assignment 6: Text classification using Deep Learning

The goal of the assignment is to classify which season a line from the tv show Game of Thrones belongs to. I order to to this, I employ both a multinomial logistic regression using a countvectorizer to create the model input, and a convolutional neural network using word embeddings. In order to prevent overfitting in the cnn, different regularization methods are included as command-line options. The script outputs confusion matrix heatmaps, a classification report in .csv format, and for the cnn, also a history plot.

### How to run

To run this code, please follow the guide for activating the virtual environment in [cds-language-exam](https://github.com/Guscode/cds-language-exam).

To test the script, in the virtual environment, please run:
```bash
cd Assignment_6
python GOT_classification.py --input-path data/Game_of_Thrones_Script.csv --output output --epochs 10 --names include
```
This will return results from both logistic regression and cnn. In order to test the cnn with regularization, run:

```bash
python GOT_classification.py --input-path data/Game_of_Thrones_Script.csv --output output --epochs 10 --names include --model cnn --dropout True --regularizer 1e-7
```

# User defined arguments

The user defined arguments are:

```bash
--input-path #Path to GOT script
--output #Path where you want the output files
--test_split # size of test data, default = 0.2
--reqularizer # float indicating regularization coefficient, default = 1e-3
--epochs #Specifies amount of epochs for the cnn
--model # Specify which model(s) to run. default = both, options: lr, cnn
--dropout # add dropout layer to the cnn
```

# Methods

To solve this assignment, the script GOT_classification.py was made. The script takes the Game of Thrones script and performs a classification task using a multinomial logistic regression and a convolutional neural network. Before fitting the data to a multinomial logistic regression, some preprocessing has to be done. Firstly, the script defines the text and the labels. In this step, a command line argument was created that allows the user to either include or exclude the name of the speaker in the text. This was done in order to investigate the predictive value of speaker, which is hypothesized to be high, as Game of Thrones is known for killing off characters.
The input for the multinomial logistic regression is a sparse document-term matrix which is produced using Scikit-Learn’s function CountVectorizer (Pedregosa et al., 2011), which creates a document-term matrix from the entire script. The input is then scaled, and the multinomial logistic regression is fitted to the training set. 
For the convolutional neural network, however, labels are binarized and the texts are tokenized. Then, the texts are padded to have the same length, and represented as an embedding matrix using 50 dimensions of GloVe embeddings (Pennington et al., 2014). Thus, the first layer is an embedding layer, followed by a convolution layer. Then a max pooling layer is employed, and before the dense layers, a command line option to include a dropout layer is built into the script. Similarly, the user is also able to specify the L2 regularization. Both adding dropout layers and lowering the regularization coefficient will prevent overfitting. If no regularization was included, the model would quickly reach 100% accuracy on the training data, as there are more features than documents. Adding L2 regularization performs ridge regression, which creates a less complex model when many features are included. Similarly, adding the dropout layer randomly drops 20%, which adds effective regularization as well. After the model is compiled, it is fitted to the training data with the number of epochs specified in the command line. The model is then used to predict the test data and the results are saved.

# Discussion
Figure 1:
<a href="https://github.com/Guscode/cds-language-exam">
    <img src="/assignment_6/output/history_plot_cnn.jpg" alt="Logo" width="600" height="600">
</a>

figure 2:
history_plot_cnn (1).jpg![image](https://user-images.githubusercontent.com/35924673/119662170-5db8d000-be31-11eb-98ad-c0f5969f61ba.png)


In order to evaluate the ability to predict the Game of Thrones season from a line, the model was run with the speaker excluded and included, and one with names included but with dropout layer and increased regularization. From table 1 it is clearly seen that including name of the speaker in the text clearly improves the models. Curiously, the multinomial logistic regression is the best model, which might be due to the simpler representation of texts, and especially regarding speaker names and entities in the text, which are not found in the GloVe embeddings. Figure X and X shows the loss and accuracy on training and validation through 10 epochs, clearly indicating overfitting, as the validation accuracy quickly stagnates, while the training accuracy keeps increasing. However, figure X shows a slower increase in training accuracy, indicating that the regularization is working. The fact that this does not amount to a greater validation accuracy can be caused by multiple things. Firstly, there is a chance that the model is overfitting to a signal in the training data which is not predictive of season number. Secondly, even though the data is flawless in terms of labelling, there might not be enough signal in the text to identify the seasons, deeming it an impossible task. In this specific dataset, identical lines appear in different seasons, making it impossible for the model to distinguish. A way to overcome this problem would be feature extraction or text extension. Feature extraction could be adding information about both the speaker and entities in the text. Concatenating subsequent lines would be a way of using text extension to enhance the information level in each data point, which can be beneficial despite halving the number of datapoints. Overall, the models perform much better than chance, indicating a degree of predictability, but the models do not perform well enough to be considered useful.

Table 1
![image](https://user-images.githubusercontent.com/35924673/119661524-acb23580-be30-11eb-8d2f-6570da5356db.png)


