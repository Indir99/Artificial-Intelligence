
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
# ZA confusion matricu
from sklearn.metrics import confusion_matrix


# UCITAVANJE DATASET-A
data = pd.read_csv("Cable-Production-Line-Dataset.csv")

# ISPISIVANJE INFORMACIJA O DATASET-U
# print(data.info())

# ISPISIVANJE DIMENZIJA DATASET-A
print("DATASET dimensions before drop command: ")
print(data.shape)

# BRISEMO DATUM JER JE TIPA OBJECT. S OVIM SMO IZBJEGLI KONVERZIJU TIPA PODATAKA
# Date kolona nam nije toliko bitna za realizaciju ovog zadatka
del data['Date']
data.drop(data[data['Cable Failures'] == 0].index, inplace=True)

# ISPISIVANJE DIMENZIJA DATASET-A
print("DATASET dimensions after drop command: ")
print(data.shape)


X = data[['Machine', 'Shift', 'Operator', 'Cable Failures',
         'Cable Failure Downtime', 'Other Failures', 'Other Failure Downtime']]

y = data['Cable Failures']
print(X.shape, y.shape)

# UKOLIKO U KOLONI IMAMO TEKSTUALNE VRIJEDNOSTI TREBA IH PRETVORITI U NUMERIČKE
data['Shift'] = [0 if each == "A" else 1 for each in data['Shift']]

x = data[['Machine', 'Shift', 'Operator', 'Cable Failure Downtime',
          'Other Failures', 'Other Failure Downtime']]

# PREBACIVANJE VRIJEDNOSTI U MATRICU ZA PREDIKCIJU (X) I VEKTOR CILJNIH VRIJEDNOSTI (Y) (PANDAS TABELA u NUMPY MATRICU)
x = x.values
y = y.values.ravel()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
# test_size=0.2 means %20 test datas, %80 train datas
method_names = []
method_scores = []
# These are for barplot in conclusion

# KNN n_neighbors=2
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
print("Score for Number of Neighbors= 2 : {}".format(knn.score(x_test, y_test)))
method_names.append("KNN")

method_scores.append(knn.score(x_test, y_test))
# Confusion Matrix
y_pred = knn.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
# Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5,
            linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


# KNN n_neighbors=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
print("Score for Number of Neighbors= 5 : {}".format(knn.score(x_test, y_test)))
method_names.append("KNN")

method_scores.append(knn.score(x_test, y_test))
# Confusion Matrix
y_pred = knn.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
# Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5,
            linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()

# KNN n_neighbors=7
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
print("Score for Number of Neighbors= 7 : {}".format(knn.score(x_test, y_test)))
method_names.append("KNN")

method_scores.append(knn.score(x_test, y_test))
# Confusion Matrix
y_pred = knn.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
# Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5,
            linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()

# SVC klasifikacija kernel='linear'
svm = SVC(random_state=42, kernel='linear', probability=True)
svm.fit(x_train, y_train)
print("SVM Classification Score for linear kernel is : {}".format(
    svm.score(x_test, y_test)))
method_names.append("SVM")
method_scores.append(svm.score(x_test, y_test))
# Confusion Matrix
y_pred = svm.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
# Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5,
            linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()

# SVC klasifikacija kernel='poly'
svm = SVC(random_state=42, kernel='poly', probability=True)
svm.fit(x_train, y_train)
print("SVM Classification Score for poly kernel is : {}".format(
    svm.score(x_test, y_test)))
method_names.append("SVM")
method_scores.append(svm.score(x_test, y_test))
# Confusion Matrix
y_pred = svm.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
# Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5,
            linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()

# SVC klasifikacija kernel='rbf'
svm = SVC(random_state=42, kernel='rbf', probability=True)
svm.fit(x_train, y_train)
print("SVM Classification Score for rbf kernel is : {}".format(
    svm.score(x_test, y_test)))
method_names.append("SVM")
method_scores.append(svm.score(x_test, y_test))
# Confusion Matrix
y_pred = svm.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
# Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5,
            linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()


naive_bayes = GaussianNB()
#naive_bayes = ComplementNB()
#naive_bayes = MultinomialNB()
#naive_bayes = BernoulliNB()
naive_bayes.fit(x_test, y_test)
print("Naive Bayes Classification (GaussianNB) Score : {} ".format(
    naive_bayes.score(x_test, y_test)))
method_names.append("Naive Bayes")

method_scores.append(naive_bayes.score(x_test, y_test))
# Confusion Matrix
y_pred = naive_bayes.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
# Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5,
            linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()

#naive_bayes = GaussianNB()
naive_bayes = ComplementNB()
#naive_bayes = MultinomialNB()
#naive_bayes = BernoulliNB()
naive_bayes.fit(x_test, y_test)
print("Naive Bayes Classification (ComplementNB) Score : {} ".format(
    naive_bayes.score(x_test, y_test)))
method_names.append("Naive Bayes")

method_scores.append(naive_bayes.score(x_test, y_test))
# Confusion Matrix
y_pred = naive_bayes.predict(x_test)
conf_mat = confusion_matrix(y_test, y_pred)
# Visualization Confusion Matrix
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(conf_mat, annot=True, linewidths=0.5,
            linecolor="red", fmt=".0f", ax=ax)
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
