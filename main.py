import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import scipy.stats as stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from ydata_profiling import ProfileReport
import mlflow
from datetime import datetime
df = pd.read_csv(r"C:\Users\dell\Downloads\Consumo_cerveja.csv")
print(df.head())

print(df.shape)
#profile = ProfileReport(df, title="Pandas Profiling Report")
#profile.to_file("your_report.html")

df["Temperatura Media (C)"]=df["Temperatura Media (C)"].str.replace(",",".").astype(float)
df["Temperatura Minima (C)"]=df["Temperatura Minima (C)"].str.replace(",",".").astype(float)
df["Temperatura Maxima (C)"]=df["Temperatura Maxima (C)"].str.replace(",",".").astype(float)
df["Precipitacao (mm)"]=df["Precipitacao (mm)"].str.replace(",",".").astype(float)

print(df.to_string())
print(df.info())
df['Data'] = pd.to_datetime(df['Data'])
df['Month']=df.Data.dt.month
df['day']=df.Data.dt.dayofweek
df.set_index('Data',inplace=True)
#print(df.to_string())
print(df.isnull().sum())
print(df.shape)
print(df.isnull().all(axis=1).sum())
df.dropna(how='all',inplace=True)
print(df.to_string())
print(df.isnull().sum())
if df.duplicated().any():
    print("there are duplicated values")
else:
    print("No duplicate values")
df["Final de Semana"]=df["Final de Semana"].astype(int)
print(df.info())
df.describe()
"""df.boxplot(figsize=(15,18))
plt.show()"""
"""df['Precipitacao (mm)'].hist(bins=100,figsize=(10,10))
plt.show()"""
print(df['Precipitacao (mm)'][df['Precipitacao (mm)']==0].value_counts())
df['Precipitacao (mm)'] = np.clip(df['Precipitacao (mm)'], 0, 40)
"""df['Precipitacao (mm)'].hist(bins=100,figsize=(10,10))
plt.show()"""
"""column_data = df['Precipitacao (mm)']

# Create a probability plot
stats.probplot(column_data, plot=plt)

# Set plot title and labels
plt.title("Normal Probability Plot")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Sample Quantiles")

# Show the plot
plt.show()"""
"""df.boxplot(figsize=(15,18))
plt.show()"""
plt.figure(figsize=(20, 18))
correlation = df.corr()
sns.heatmap(correlation, annot = True)
X= df.drop(columns=["Consumo de cerveja (litros)"],axis=1)
y= df["Consumo de cerveja (litros)"]
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.20,random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
lr = LinearRegression()
lr.fit(X_train,y_train)
print('The final coefficients after training is:',lr.coef_)
print('The final intercept after training is:',lr.intercept_)
y_pred = lr.predict(X_test)
print("r2 score of our model is:", r2_score(y_test,y_pred))
print("mean absolute error of our model is:", mean_absolute_error(y_test,y_pred))
print("root mean squared error of our model is:", mean_squared_error(y_test,y_pred,squared=False))
new_data = np.array([[25.0, 20.0, 30.0, 5.0, 1, 9, 15],
                     [30.0, 22.0, 32.0, 10.0, 0, 10, 4],
                     [28.0, 18.0, 33.0, 0.0, 1, 11,7]])


X_new = pd.DataFrame(new_data, columns=["Temperatura Media (C)", "Temperatura Minima (C)", "Temperatura Maxima (C)", "Precipitacao (mm)", "Final de Semana", "Month", "day"])
y_pred = lr.predict(X_new)
print(y_pred)