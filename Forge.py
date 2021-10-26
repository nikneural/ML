import mglearn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# from sklearn.neighbors import KNeighborsClassifier
#
#
X, y = mglearn.datasets.make_forge()
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend(["Класс 0", "Класс 1"], loc=4)
# plt.xlabel("Первый признак")
# plt.ylabel("Второй признак")
# print("Форма массива X: {}".format(X.shape))
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# clf = KNeighborsClassifier(n_neighbors=3)
# clf.fit(X_train, y_train)
#
# print("Прогноы на тестовом наборе: {}".format(clf.predict(X_test)))
# print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test))

fig, axes = plt.subplots(1, 2, figsize=(10, 3))

for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Признак 0")
    ax.set_ylabel("Признак 1")
axes[0].legend()
plt.show()




















