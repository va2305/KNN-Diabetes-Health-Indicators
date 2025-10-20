#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#to split dataset
from sklearn.model_selection import train_test_split

#to scale features
from sklearn.preprocessing import StandardScaler

#KNN algo 
from sklearn.neighbors import KNeighborsClassifier

#to evaluate model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#load Dataset
data = pd.read_csv(r"diabetes_012_health_indicators_BRFSS2015.csv")
print(data.head())


#check shape of dataset
print("\n Dataset shape(rows,columns):", data.shape)

#check column names
print("\n Column names:")
print(data.columns)

#check basic info(data types, nulls)
print("\n Dataset info:")
print(data.isnull().sum())

#check class distribution
print("\n Target column distribution:")
print(data['Diabetes_012'].value_counts())


#Visualize 
sns.countplot(x='Diabetes_012', data=data)
plt.title("Target Class Distribution (0 = No Diabetes, 1 = Prediabetes, 2 = Diabetes)")
plt.show()

#---------Feature & Target split----------#

#Separate Indep features(X) & target variable(y)
X = data.drop('Diabetes_012', axis= 1)
y = data['Diabetes_012']

print("\n Features and Target separated")
print("Feature shape:", X.shape)
print("Feature shape:", y.shape)

#-------Train test Split------#
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify= y)

print("\n Data split done:")
print("\n Training set:", X_train.shape)
print("Test set:", X_test.shape )

#--------Feature Scaling-------#
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n Feature scaling completed.")


#-------Model Training & Prediction-------#
#Initialize KNN Classifier, start K = 5
knn = KNeighborsClassifier(n_neighbors=5)

#Train the model on Training data
knn.fit(X_train_scaled, y_train)

print("\n KNN model trained successfully")

#Predict test set
y_pred = knn.predict(X_test_scaled)

print("\n Prediction sample:")
print(y_pred[:20])


#--------Model Evaluation-------#
#Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n Model Accuracy: {accuracy*100:.2f}%")

#Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n Confusion matrix:\n", cm)

#Plot confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt= 'd', cmap= 'blues')
plt.title("Confusion matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


#Classification Report
print("\n Classification Repot:\n")
print(classification_report(y_test, y_pred))