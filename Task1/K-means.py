from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ZA confusion matricu
from sklearn.metrics import confusion_matrix

# UCITAVANJE DATASET-A
data = pd.read_csv("Cable-Production-Line-Dataset.csv")

# ISPISIVANJE INFORMACIJA O DATASET-U
print(data.info())

# ISPISIVANJE DIMENZIJA DATASET-A
print(data.shape)

# ISPISIVANJE PRVIH 5 REDOVA DATASET-A
print(data.head())

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


# UKOLIKO U KOLONI IMAMO TEKSTUALNE VRIJEDNOSTI TREBA IH PRETVORITI U NUMERIČKE
data['Shift'] = [0 if each == "A" else 1 for each in data['Shift']]

# ---------------------------------------------------------------------------------------------------
# K-means za prozivoljno odabrane dvije kolone
#
#x = data[['Operator', 'Cable Failure Downtime']]
# print(x.head())

# PREBACIVANJE VRIJEDNOSTI U MATRICU ZA PREDIKCIJU (X) (PANDAS TABELA u NUMPY MATRICU)
#x = x.values

# Potrebno uzeti razlicit broj klastera, te na osnovu toga vrsiti analizu
# Testirati sa: 2,4,8,12
#model = KMeans(n_clusters=12)
# fit the model
# model.fit(x)
# assign a cluster to each example
#yhat = model.predict(x)
# retrieve unique clusters
#clusters = np.unique(yhat)
# create scatter plot for samples from each cluster
# for cluster in clusters:
# get row indexes for samples with this cluster
#    row_ix = np.where(yhat == cluster)
# create scatter of these samples #mijenjati brojeve 2 i 3 da se vide različiti atributi na 2D plotu
#    plt.scatter(x[row_ix, 0], x[row_ix, 1])
# show the plot
# plt.show()

# -------------------------------------------------------------------------------------------------------
#
x = data[['Machine', 'Shift', 'Operator', 'Cable Failures',
          'Cable Failure Downtime', 'Other Failures', 'Other Failure Downtime']]
print(x.head())

# PREBACIVANJE VRIJEDNOSTI U MATRICU ZA PREDIKCIJU (X) (PANDAS TABELA u NUMPY MATRICU)
x = x.values

# Potrebno uzeti razlicit broj klastera, te na osnovu toga vrsiti analizu
# Testirati sa: 2,4
model = KMeans(n_clusters=4)
# fit the model
model.fit(x)
# assign a cluster to each example
yhat = model.predict(x)
# retrieve unique clusters
clusters = np.unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = np.where(yhat == cluster)
# create scatter of these samples
# mijenjati brojeve kolona da se vide različiti atributi na 2D plotu
    plt.scatter(x[row_ix, 4], x[row_ix, 6])
# show the plot
plt.show()
# ---------------------------------------------------------------------------------------------------
