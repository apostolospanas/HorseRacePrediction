# HorseRacePrediction
Holistic Quantitative Benchmarking of Machine Learning Algorithms: Regression, Classification, and Beyond for Horse Race Predictions

# Horse Racing Data Analysis and Prediction

This project involves the analysis and prediction of horse racing data using various machine learning models. The dataset includes historical records of horse races and aims to predict outcomes such as win probability, finish positions, and prize money. The project leverages models like Random Forest, Decision Tree, Neural Networks, and XGBoost to achieve these predictions.

## Table of Contents
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Evaluation](#evaluation)
- [Feature Engineering](#feature-engineering)
- [Cross-Validation](#cross-validation)
- [Visualization](#visualization)
- [License](#license)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need to have the following libraries and tools installed:

- Python 3.7+
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- Tqdm
- XGBoost
- TensorFlow
- Psutil

### Installation

Clone the repository to your local machine and navigate to the project directory:

```sh
git clone https://github.com/yourusername/horse-racing-prediction.git
cd horse-racing-prediction
Install the required Python packages:

sh
Αντιγραφή κώδικα
pip install -r requirements.txt
Usage
Ensure the dataset is downloaded and extracted into the horse_racing_data directory:

sh
Αντιγραφή κώδικα
kaggle datasets download -d hwaitt/horse-racing
unzip horse-racing.zip -d horse_racing_data
Run the analysis script:

sh
Αντιγραφή κώδικα
python analysis.py
Models Used
Linear Regression
Logistic Regression
Random Forest Regressor
Random Forest Classifier
MLP Regressor (Neural Network)
MLP Classifier (Neural Network)
XGBoost Classifier
Naive Bayes
Decision Tree Regressor
Decision Tree Classifier
K-Nearest Neighbors Regressor
K-Nearest Neighbors Classifier
Evaluation
The models are evaluated based on various metrics like Mean Squared Error (MSE), Accuracy, ROC AUC score, and more. The results are then visualized using plots to compare the performance of different models.

Feature Engineering
Feature engineering steps include:

Converting distances to a unified format
Calculating average speed
Encoding categorical variables
Handling missing values
Cross-Validation
We use cross-validation techniques to ensure the robustness of our models. This includes shuffling and splitting the dataset into training and testing sets multiple times to validate the performance.

Visualization
Several visualizations are provided to understand the data and model performance better, including:

Distribution plots
Pairwise relationship plots
ROC curve
Learning curves
Confusion matrix
Residual plots
Precision-Recall curve
Feature importance plots
Heatmaps
License
This project is licensed under the MIT License - see the LICENSE file for details.
