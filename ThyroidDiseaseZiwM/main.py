import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.model_selection import cross_validate, train_test_split
from algorithms import find_best_features
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allbp.data"
url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allhyper.data"

# Assign colum names to the dataset

# Read dataset to pandas dataframe
data = pd.read_csv("dataset.csv")

X = data.iloc[:, :23]
Y = data.iloc[:, -1]

fvalue_selector = SelectKBest(f_classif)
fvalue_selector.fit(X, Y)
rank = fvalue_selector.scores_
print(rank)
top_rank = []
indexes = rank.argsort()[-10:][::-1]

for index in indexes:
    top_rank.append(rank[index])


best_score = [0, '', 0, 0, np.ndarray]
list_score = []

neighbors = [1, 5, 10]
distances = ['euclidean', 'manhattan']


# for neighbor in neighbors:
#     for distance in distances:
#         for feature_number in range(1, 10):
#             best_score = class
# '



