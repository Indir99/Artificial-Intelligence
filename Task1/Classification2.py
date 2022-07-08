import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# UCITAVANJE DATASET-A
data = pd.read_csv("Cable-Production-Line-Dataset.csv")

# ISPISIVANJE INFORMACIJA O DATASET-U
# print(data.info())

# ISPISIVANJE DIMENZIJA DATASET-A
print("DATASET dimensions before drop command: ")
print(data.shape)

# BRISANJE REDOVA KOD KOJIH JE CABLE FAILURES == 0
data.drop(data[data['Cable Failures'] == 0].index, inplace=True)

# ISPISIVANJE DIMENZIJA I INFORMACIJA POSLIJE DROP KOMANDE
# print(data.info())
# ISPISIVANJE DIMENZIJA DATASET-A
print("DATASET dimensions after drop command: ")
print(data.shape)
