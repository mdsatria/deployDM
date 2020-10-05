# import library
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=12)

# Eksperimen dengan mengubah parameter jumlah neighbours
acc_score = []
for i in ([1, 3, 5, 7, 9, 11]):
    clf = KNeighborsClassifier(n_neighbors=i)
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    acc_score.append([i, acc])
print(acc_score)

# Train dengan parameter terbaik
clf = KNeighborsClassifier(n_neighbors=11)
clf.fit(x_train, y_train)

# Save model dengan parameter terbaik
with open("model.pkl", "wb") as file:
    pickle.dump(clf, file)
