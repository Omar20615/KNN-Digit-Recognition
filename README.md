# KNN-Digit-Recognition

This project aims to classify hand-written digits ('3', '8', and '6') using the K-Nearest Neighbors (KNN) algorithm. The project consists of two main tasks: a basic implementation of KNN for digit recognition and a more advanced version that enhances accuracy through custom feature extraction techniques.

KNN_Digit_Classifier.py
This script implements a basic KNN classifier for hand-written digit classification. The digits '3', '8', and '6' are extracted from the USPS dataset and used to create a training set. The key steps include:

Custom Distance Calculation: A Euclidean distance function is used to calculate distances between digit instances.
Neighbor Search: A custom function finds the nearest neighbors for each digit instance.
Accuracy Evaluation: The classifier's accuracy is calculated across different values of k (number of neighbors), and the results are visualized.
Data Splitting: The data is split into training and testing sets multiple times to observe the performance over multiple runs.
Visualization of Errors: The script includes a visualization of misclassified digits, highlighting the mistakes made by the classifier.


KNN_Digit_Feature_Extractor.py
This script builds upon the previous task by applying custom feature extraction techniques to improve the accuracy of the KNN classifier. The steps include:

Feature Extraction: A custom feature extraction pipeline is applied to the digit images. This pipeline includes 5x5 and 3x3 convolutional filters along with max pooling to capture more detailed patterns in the images.
Accuracy Analysis: The enhanced features are used to classify the digits using KNN, and the accuracy is evaluated over different k values.
Confusion Matrix: A confusion matrix is computed and visualized using a heatmap to analyze the classification performance of the model, providing insights into specific errors made during classification.
