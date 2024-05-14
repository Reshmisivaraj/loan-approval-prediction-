import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle


dataset = pd.read_csv('/content/trial_dataset.csv')

print(dataset.head())


print(dataset.describe())

print(dataset.isnull().sum())


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


classifier = LogisticRegression()
classifier.fit(X_train, y_train)

threshold = 0.6


prediction_threshold_adjusted = (classifier.predict_proba([[3, 1, 0, 9100000, 29700000, 20, 578, 71000, 450000, 33300000, 12800000]])[:, 1] >= threshold).astype(int)
print("Predicted class label with adjusted threshold:", prediction_threshold_adjusted)


with open('LoanAmountPrediction.pkl', 'wb') as f:
    pickle.dump(classifier, f)


from google.colab import files
files.download('LoanAmountPrediction.pkl')


