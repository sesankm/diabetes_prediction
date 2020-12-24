import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Preprocessing
df = pd.read_csv("diabetes_data_upload.csv")
df.iloc[:, 1:] = df.iloc[:, 1:].applymap(lambda x: int(x.lower() == 'yes' or x.lower() == 'positive' or x.lower() == 'male'))
df.dropna(inplace=True)

# EDA
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), cmap='Blues')
plt.show()

class_by_gender = df.groupby("Gender").agg(sum)["class"]
plt.title("Number of people with diabetes vs Gender")
sns.barplot(x=class_by_gender.index.tolist(), y=class_by_gender.tolist())
plt.xticks([0, 1], ["Female", "Male"])
plt.show()

dia = df[df["class"] == 1]
plt.title("Age distribution for peopl with diabetes")
sns.histplot(dia["Age"], kde=True)
plt.show()

# Model Building
X = df[["Polyuria", "Polydipsia", "sudden weight loss", "partial paresis"]].to_numpy()
y = df["class"].to_numpy()
x_train, x_test, y_train, y_test = train_test_split(X, y)

svc = SVC().fit(x_train, y_train)
ridge = RidgeClassifier().fit(x_train, y_train)


nn_model = keras.Sequential([
        layers.Dense(16, activation="relu"),
        layers.Dense(12, activation="relu"),
        layers.Dense(12, activation="tanh"),
        layers.Dense(8, activation="tanh"),
        layers.Dense(4, activation="tanh"),
        layers.Dense(1, activation="sigmoid")
])
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
nn_model.compile(optimizer=adam, loss=tf.keras.losses.MeanSquaredError())
nn_model.fit(x=x_train, y=y_train, epochs=500)

print("SVC MSE: {}".format(mean_squared_error(y_test, svc.predict(x_test))))
print("Ridge Classifier MSE:{}".format(mean_squared_error(y_test, ridge.predict(x_test))))
print("Neural Network MSE: {}".format(mean_squared_error(y_test, nn_model.predict(x_test))))