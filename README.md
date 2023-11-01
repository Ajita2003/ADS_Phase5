# Electricity Price Prediction using Random Forest Regressor

## Overview

This code is a Python script for predicting electricity prices using a Random Forest Regressor model. It includes data loading, preprocessing, model training, and evaluation steps. The code reads a dataset from a CSV file named "Electricity.csv" and uses the Random Forest Regressor to predict the electricity price (SMPEP2) based on various features.

## Dataset

The kaggle link for the data set is given below:
https://www.kaggle.com/datasets/chakradharmattapalli/electricity-price-prediction/

## Getting Started

To run this code, you need Python installed on your system along with the following libraries: pandas, numpy, matplotlib, seaborn, scikit-learn. You can install these libraries using pip if they are not already installed:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. **Data Loading:**
   - The code loads the dataset from a CSV file. Make sure to have your dataset in a file named "Electricity.csv" in the same directory as the script.

2. **Data Preprocessing:**
   - The code performs data preprocessing, which includes changing the data types of certain columns to numerical values and scaling the features using StandardScaler. It also cleans the data by removing rows with missing values.

3. **Data Splitting:**
   - The dataset is split into training and testing sets (80% training and 20% testing) using train_test_split from scikit-learn.

4. **Model Training:**
   - A Random Forest Regressor model is created and trained on the training data.

5. **Feature Input and Prediction:**
   - You can input feature values to predict the electricity price (SMPEP2). The features include Day, Month, ForecastWindProduction, SystemLoadEA, SMPEA, ORKTemperature, ORKWindspeed, CO2Intensity, ActualWindProduction, and SystemLoadEP2. The code will predict the price based on your input.

6. **Model Evaluation:**
   - After making a prediction, you can enter the actual electricity price. The code will calculate and display Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) to evaluate the model's performance.

## Customization

You can modify the code to work with your own dataset by changing the file name or by adjusting the features used for prediction.

## License

This code is provided under an open-source license. You are free to use and modify it for your purposes.

## Authors

1. Ajita Fairen J - 715521106003 - B.E. ECE - 3rd year
2. M. Mekanya - 715521106028 - B.E. ECE - 3rd year
3. Shanmugapriya M - 715521106043 - B.E. ECE - 3rd year
4. K Shree Harini - 715521106044 - B.E. ECE - 3rd year
5. Vasanth L - 715521106055 - B.E. ECE - 3rd year
6. G. R. Tharunika - 715521106310 - B.E. ECE - 3rd year
 
