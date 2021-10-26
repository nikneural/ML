import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
# print("Ключи cancer: {}".format(cancer.keys()))
# print("Форма массива data для набора cancer: {}".format(cancer["data"].shape))
# print("Количество примеров для каждого класса:\n{}".format({n: v for n, v in zip(cancer['target_names'], np.bincount(cancer['target']))}))
# print("Имена признаков:\n{}".format(cancer['feature_names']))

X_train, X_test, y_train, y_test = train_test_split(cancer['data'], cancer['target'], stratify=cancer['target'],random_state=42)
#
# training_accuracy = []
# test_accuracy = []

# neighbors_setting = range(1, 11)
#
# for n_neighbors in neighbors_setting:
#     clf = KNeighborsClassifier(n_neighbors=n_neighbors)
#     clf.fit(X_train, y_train)
#     training_accuracy.append(clf.score(X_train, y_train))
#     test_accuracy.append(clf.score(X_test, y_test))
#
# plt.plot(neighbors_setting, training_accuracy, label='правильность на обучающем наборе')
# plt.plot(neighbors_setting, test_accuracy, label='правильность на тестовом наборе')
# plt.ylabel("Правильность")
# plt.xlabel("количество соседей")
# plt.legend()
# plt.show()

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(logreg.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(logreg.score(X_test, y_test)))

logreg100 = LogisticRegression(C=100)
logreg100.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(logreg100.score(X_test, y_test)))

logreg001 = LogisticRegression(C=0.01)
logreg001.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(logreg001.score(X_test, y_test)))
