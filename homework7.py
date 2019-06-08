import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


diabetes=load_diabetes()

x=diabetes.data
y=diabetes.target

print(f'diabetes of x shape:{x.shape}, diabetes of y shape:{y.shape} ')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

# Training a Linear Regression model with fit()

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(x_train, y_train)

# Predicting the results for our test dataset

predicted_values = lm.predict(x_test)

# Printing the residuals: difference between real and predicted

for (real, predicted) in list(zip(y_test, predicted_values)):
    print(f"Value: {real:.2f}, pred: {predicted:.2f}, diff: {(real - predicted):.2f}")

##############
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35)

from sklearn.neighbors import KNeighborsRegressor
training_accuracy = []
test_accuracy = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn =KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(x_train, y_train)
    # record training set accuracy
    training_accuracy.append(knn.score(x_train, y_train))

    print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(x_train, y_train)))
    print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(x_test, y_test)))

    # record test set accuracy
    test_accuracy.append(knn.score(x_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.savefig('plots/knn_regressor_n_neighbors.png')

