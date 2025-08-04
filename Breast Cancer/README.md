Breast Cancer Classification Project
Overview
This project implements a Random Forest Classifier to predict breast cancer diagnosis (malignant or benign) using the Breast Cancer Wisconsin dataset from scikit-learn. The code includes data loading, inspection, model training, evaluation, and visualization of results.
Features

Loads and preprocesses the Breast Cancer Wisconsin dataset
Trains a Random Forest Classifier
Evaluates model performance using accuracy, classification report, and confusion matrix
Visualizes results with a confusion matrix heatmap and feature importance plot

Requirements
To run this project, install the required Python libraries:
pip install numpy pandas matplotlib scikit-learn seaborn

Usage

Clone the repository:

git clone <repository-url>
cd <repository-directory>


Run the Python script:

python breast_cancer_classification.py

Code Structure
The main script (breast_cancer_classification.py) follows these steps:

Import Libraries: Imports necessary libraries (numpy, pandas, matplotlib, scikit-learn, seaborn).
Load Dataset: Loads the Breast Cancer Wisconsin dataset from scikit-learn.
Inspect Dataset: Displays class distribution and a sample of features.
Split Data: Splits data into training (80%) and testing (20%) sets.
Train Model: Trains a Random Forest Classifier with 100 estimators.
Predict and Evaluate: Generates predictions and prints accuracy and classification report.
Confusion Matrix: Visualizes the confusion matrix as a heatmap.
Feature Importance: Plots the importance of each feature in the model.

Output

Console Output:
Class distribution of the dataset
Sample of feature data
Model accuracy
Detailed classification report (precision, recall, f1-score)


Visualizations:
Confusion matrix heatmap showing true vs. predicted labels
Bar plot of feature importance



Dataset
The project uses the Breast Cancer Wisconsin dataset from scikit-learn, which contains 569 samples with 30 features each. The target variable indicates whether a tumor is malignant (0) or benign (1).
License
This project is licensed under the MIT License. See the LICENSE file for details.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
