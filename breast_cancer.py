import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer

#load breast cancer values from the data set
cancer = load_breast_cancer()

#keys are the coloumns or dicts that we have, dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])
keys = cancer.keys()
# print (cancer.keys())
target_names = cancer ['target_names']
target = cancer ['target']
DESCR = cancer ['DESCR']
feature_names = cancer ['feature_names']
filename = cancer ['filename']
data = cancer ['data'].shape
print (data)
# print (target_names)
# creates a dataframe out of 'data' and 'target' and adding an additional column by appending 'feature_names' and 'target'
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))

# head helps to show the head values in rows and columns
# tail helps to show the last values in rows and columns
values_head = df_cancer.head()
values_tail = df_cancer.tail()
# print (values_head)
# print (values_tail)

# pairplot helps to show the quick glance of the entire data at one place
pairplot = sns.pairplot(df_cancer, vars = ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
print (pairplot)

sns.countplot(df_cancer['target']) #countplot shows the three dimensional surface on a two dimensional plane

sns.scatterplot(x = 'mean area', y = 'mean smoothness', hue = 'target', data = df_cancer) #scatterplot shows the data as a collection of points. The position of a point 

plt.figure(figsize = (20, 10))
sns.heatmap(df_cancer.corr(), annot = True) #heatmap representing values with correlations.


