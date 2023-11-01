import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv("Electricity.csv")

# Print the first few rows and data info
print(data.head())
data.info()

# Convert relevant columns to numeric, handling errors
numeric_columns = [
    "ForecastWindProduction", "SystemLoadEA", "SMPEA",
    "ORKTemperature", "ORKWindspeed", "CO2Intensity",
    "ActualWindProduction", "SystemLoadEP2", "SMPEP2"
]
data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Check for missing values and drop rows with missing values
data.isnull().sum()
data.dropna(inplace=True)

# Select features for correlation calculation
features = [
    "ForecastWindProduction", "SystemLoadEA", "SMPEA",
    "ORKTemperature", "ORKWindspeed", "CO2Intensity",
    "ActualWindProduction", "SystemLoadEP2", "SMPEP2"
]

# Calculate correlations and create a heatmap
correlations = data[features].corr(method='pearson')
plt.figure(figsize=(16, 12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()
