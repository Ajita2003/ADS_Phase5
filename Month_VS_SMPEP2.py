import pandas as pd
import numpy as np
data = pd.read_csv("Electricity.csv")
print(data.head())
data.info()

data["ForecastWindProduction"] = pd.to_numeric(data["ForecastWindProduction"], errors= 'coerce')
data["SystemLoadEA"] = pd.to_numeric(data["SystemLoadEA"], errors= 'coerce')
data["SMPEA"] = pd.to_numeric(data["SMPEA"], errors= 'coerce')
data["ORKTemperature"] = pd.to_numeric(data["ORKTemperature"], errors= 'coerce')
data["ORKWindspeed"] = pd.to_numeric(data["ORKWindspeed"], errors= 'coerce')
data["CO2Intensity"] = pd.to_numeric(data["CO2Intensity"], errors= 'coerce')
data["ActualWindProduction"] = pd.to_numeric(data["ActualWindProduction"], errors= 'coerce')
data["SystemLoadEP2"] = pd.to_numeric(data["SystemLoadEP2"], errors= 'coerce')
data["SMPEP2"] = pd.to_numeric(data["SMPEP2"], errors= 'coerce')
data.isnull().sum()
data = data.dropna()


x = data[["Day", "Month", "ForecastWindProduction", "SystemLoadEA", "SMPEA", "ORKTemperature", "ORKWindspeed", "CO2Intensity", "ActualWindProduction", "SystemLoadEP2"]]
y = data["SMPEP2"]
import seaborn as sns
import matplotlib.pyplot as plt
# Assuming 'data' is your DataFrame
sns.barplot(data=data, x="Month", y="SMPEP2")

# Optionally, you can set additional parameters or customize the plot
plt.xlabel("Month")
plt.ylabel("SMPEP2")
plt.title("Month Vs SMPEP2")
plt.show()
