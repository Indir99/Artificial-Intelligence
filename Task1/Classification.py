from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_wine
from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# ZA confusion matricu
from sklearn.metrics import confusion_matrix


# UCITAVANJE DATASET-A
data = pd.read_csv("Cable-Production-Line-Dataset.csv")

# ISPISIVANJE INFORMACIJA O DATASET-U
print(data.info())

# ISPISIVANJE DIMENZIJA DATASET-A
print(data.shape)

# ISPISIVANJE PRVIH 5 REDOVA DATASET-A
# print(data.head())

# PLOTANJE VRIJEDNOSTI U IZ ODREĐENIH KOLONA
# data['Machine'].value_counts().plot(kind='bar')
# plt.show()
# data['Operator'].value_counts().plot(kind='bar')
# plt.show()
# data['Shift'].value_counts().plot(kind='bar')
# plt.show()

# BRISEMO DATUM JER JE TIPA OBJECT. S OVIM SMO IZBJEGLI KONVERZIJU TIPA PODATAKA
# Date kolona nam nije toliko bitna za realizaciju ovog zadatka
del data['Date']


X = data[['Machine', 'Shift', 'Operator', 'Cable Failures',
         'Cable Failure Downtime', 'Other Failures', 'Other Failure Downtime']]

data['Other Failures'] = [0 if each ==
                          0 else 1 for each in data['Other Failures']]
y = data['Other Failures']
print(X.shape, y.shape)

# UKOLIKO U KOLONI IMAMO TEKSTUALNE VRIJEDNOSTI TREBA IH PRETVORITI U NUMERIČKE
data['Shift'] = [0 if each == "A" else 1 for each in data['Shift']]


x = data[['Machine', 'Shift', 'Operator', 'Cable Failures',
          'Cable Failure Downtime', 'Other Failure Downtime']]


# PREBACIVANJE VRIJEDNOSTI U MATRICU ZA PREDIKCIJU (X) I VEKTOR CILJNIH VRIJEDNOSTI (Y) (PANDAS TABELA u NUMPY MATRICU)
x = x.values
y = y.values.ravel()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
# test_size=0.2 means %20 test datas, %80 train datas
method_names = []
method_scores = []
# These are for barplot in conclusion


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
print("Score for Number of Neighbors= 3: {}".format(knn.score(x_test, y_test)))
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

# ROC za KNN
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

knn_cv = KNeighborsClassifier(n_neighbors=3)
y_scores = knn.predict_proba(x_test)
fpr, tpr, threshold = roc_curve(y_test, y_scores[:, 1])
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()

# SVC klasifikacija
svm = SVC(random_state=42, kernel='linear', probability=True)
svm.fit(x_train, y_train)
print("SVM Classification Score is : {}".format(svm.score(x_test, y_test)))
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

# ROC za SVM
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, random_state=42, test_size=0.2)
# svc = SVC(random_state=42, probability=True, kernel='linear')
# #svc.fit(x_train, y_train)

# y_scores = svc.fit(x_train, y_train).decision_function(x_test)
# fpr, tpr, thresholds = roc_curve(y_test, y_scores[:])
# roc_auc = auc(fpr, tpr)

# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1.05], [0, 1.05], 'r--')
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.title('ROC Curve of SVC')
# plt.show()


naive_bayes = GaussianNB()
naive_bayes.fit(x_test, y_test)
print("Naive Bayes Classification Score:{}".format(
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

# ROC za Bayes-a
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# clf2 = GaussianNB()
# clf2.fit(x_test, y_test)
# probas = clf2.predict_proba(x_test)
# fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
# # Do not change this code! This plots the ROC curve.
# # Just replace the fpr and tpr above with the values from your roc_curve
# plt.plot(fpr, tpr, label='NB')  # plot the ROC curve
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('ROC Curve Naive Bayes')
# plt.show()
# print(roc_auc_score(y_test, probas[:, 1]))
