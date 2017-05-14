import pandas
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def encode_target(df, target_column):
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod[target_column] = df_mod[target_column].replace(map_to_int)
    return df_mod, targets


data = pandas.read_csv("graphs_info.csv")
pandas.set_option('precision', 4)

data = data.drop(['graph_name', 'radius', 'diameter', 'best_score'], 1)
data, algo_names = encode_target(data, 'best_algorithm')
print(algo_names)

X = data.drop(['best_algorithm'], 1)
y = data['best_algorithm']

# normalize data
X = (X - X.mean(axis=0)) / X.std()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

nnb = [i for i in range(ceil(sqrt(data.shape[0]))) if i % 2 != 0]
cv_scores = []
# perform 10-fold cross validation
for k in nnb:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = nnb[MSE.index(min(MSE))]
print("The optimal number of neighbors is {}".format(optimal_k))

# plot misclassification error vs k
plt.plot(nnb, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
# plt.show()

grid = GridSearchCV(KNeighborsClassifier(),
                    param_grid={'n_neighbors': nnb},
                    scoring='accuracy',
                    cv=10).fit(X_train, y_train)

best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
print(best_cv_err, best_n_neighbors)

models = {
    'rfc': RandomForestClassifier(n_estimators=1000).fit(X_train, y_train),
    'gbc': GradientBoostingClassifier(n_estimators=1000).fit(X_train, y_train),
    'knn': KNeighborsClassifier(n_neighbors=best_n_neighbors, weights='distance').fit(X_train, y_train),
    'dtc': DecisionTreeClassifier().fit(X_train, y_train),
    'lda': LinearDiscriminantAnalysis().fit(X_train, y_train),
    'qda': QuadraticDiscriminantAnalysis().fit(X_train, y_train),
    'log': LogisticRegression().fit(X_train, y_train),
    'svc': SVC().fit(X_train, y_train)
}

train_errors = {}
test_errors = {}

for model_name, model in models.items():
    train_errors[model_name] = 1 - accuracy_score(y_train, model.predict(X_train))
    test_errors[model_name] = 1 - accuracy_score(y_test, model.predict(X_test))

fig, axs = plt.subplots(ncols=2, figsize=(10, 3))

pandas.Series(train_errors, index=train_errors.keys()).plot(kind='barh', ax=axs[0])
axs[0].set_title('Train error')

pandas.Series(test_errors, index=test_errors.keys()).plot(kind='barh', ax=axs[1])
axs[1].set_title('Test error')
plt.show()

print(accuracy_score(y_test, models['knn'].predict(X_test)))
