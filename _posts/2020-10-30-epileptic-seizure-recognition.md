---
layout:     post
title:      Epileptic Seizure Detection
date:       2020-10-30 12:31:19
summary:    Using ML ensemble algorithms to classify which patients are prone to a seizure.
categories: jekyll pixyll
---


```python
%%html
<style>
@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d|ice|');
body {background-color: gainsboro;}
a {color: #37c9e1; font-family: 'Roboto';}
h1 {color: #37c9e1; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;}
h2, h3 {color: slategray; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;}
h4 {color: #818286; font-family: 'Roboto';}
span {font-family:'Roboto'; color:black; text-shadow: 5px 5px 5px #aaa;}  
div.output_area pre{font-family:'Roboto'; font-size:110%; color:lightblue;}      
</style>
```


<style>
@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d|ice|');
body {background-color: gainsboro;}
a {color: #37c9e1; font-family: 'Roboto';}
h1 {color: #37c9e1; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;}
h2, h3 {color: slategray; font-family: 'Orbitron'; text-shadow: 4px 4px 4px #aaa;}
h4 {color: #818286; font-family: 'Roboto';}
span {font-family:'Roboto'; color:black; text-shadow: 5px 5px 5px #aaa;}  
div.output_area pre{font-family:'Roboto'; font-size:110%; color:lightblue;}      
</style>




```python
import numpy as np
import pandas as pd
```


```python
df = pd.read_csv('../input/epileptic-seizure-recognition/Epileptic Seizure Recognition.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed</th>
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>X4</th>
      <th>X5</th>
      <th>X6</th>
      <th>X7</th>
      <th>X8</th>
      <th>X9</th>
      <th>...</th>
      <th>X170</th>
      <th>X171</th>
      <th>X172</th>
      <th>X173</th>
      <th>X174</th>
      <th>X175</th>
      <th>X176</th>
      <th>X177</th>
      <th>X178</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>X21.V1.791</td>
      <td>135</td>
      <td>190</td>
      <td>229</td>
      <td>223</td>
      <td>192</td>
      <td>125</td>
      <td>55</td>
      <td>-9</td>
      <td>-33</td>
      <td>...</td>
      <td>-17</td>
      <td>-15</td>
      <td>-31</td>
      <td>-77</td>
      <td>-103</td>
      <td>-127</td>
      <td>-116</td>
      <td>-83</td>
      <td>-51</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>X15.V1.924</td>
      <td>386</td>
      <td>382</td>
      <td>356</td>
      <td>331</td>
      <td>320</td>
      <td>315</td>
      <td>307</td>
      <td>272</td>
      <td>244</td>
      <td>...</td>
      <td>164</td>
      <td>150</td>
      <td>146</td>
      <td>152</td>
      <td>157</td>
      <td>156</td>
      <td>154</td>
      <td>143</td>
      <td>129</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>X8.V1.1</td>
      <td>-32</td>
      <td>-39</td>
      <td>-47</td>
      <td>-37</td>
      <td>-32</td>
      <td>-36</td>
      <td>-57</td>
      <td>-73</td>
      <td>-85</td>
      <td>...</td>
      <td>57</td>
      <td>64</td>
      <td>48</td>
      <td>19</td>
      <td>-12</td>
      <td>-30</td>
      <td>-35</td>
      <td>-35</td>
      <td>-36</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>X16.V1.60</td>
      <td>-105</td>
      <td>-101</td>
      <td>-96</td>
      <td>-92</td>
      <td>-89</td>
      <td>-95</td>
      <td>-102</td>
      <td>-100</td>
      <td>-87</td>
      <td>...</td>
      <td>-82</td>
      <td>-81</td>
      <td>-80</td>
      <td>-77</td>
      <td>-85</td>
      <td>-77</td>
      <td>-72</td>
      <td>-69</td>
      <td>-65</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>X20.V1.54</td>
      <td>-9</td>
      <td>-65</td>
      <td>-98</td>
      <td>-102</td>
      <td>-78</td>
      <td>-48</td>
      <td>-16</td>
      <td>0</td>
      <td>-21</td>
      <td>...</td>
      <td>4</td>
      <td>2</td>
      <td>-12</td>
      <td>-32</td>
      <td>-41</td>
      <td>-65</td>
      <td>-83</td>
      <td>-89</td>
      <td>-73</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 180 columns</p>
</div>




```python
df['y'].value_counts()
```




    5    2300
    4    2300
    3    2300
    2    2300
    1    2300
    Name: y, dtype: int64




```python
import seaborn as sns
```


```python
cols = df.columns
target = df.y
target.unique()
target[target > 1 ] = 0
ax = sns.countplot(target)
non_seizure, seizure = target.value_counts()
print('The number of trials for the non-seizure class is:', non_seizure)
print('The number of trials for the seizure class is:', seizure)
```

    The number of trials for the non-seizure class is: 9200
    The number of trials for the seizure class is: 2300


    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning:
    A value is trying to be set on a copy of a slice from a DataFrame

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      after removing the cwd from sys.path.



![png](epileptic-seizure-recognition_files/epileptic-seizure-recognition_5_2.png)



```python
df.isnull().sum()
```




    Unnamed    0
    X1         0
    X2         0
    X3         0
    X4         0
              ..
    X175       0
    X176       0
    X177       0
    X178       0
    y          0
    Length: 180, dtype: int64




```python
y = df.iloc[:,179].values
y
```




    array([0, 1, 0, ..., 0, 0, 0])




```python
y[y>1]=0
y
```




    array([0, 1, 0, ..., 0, 0, 0])




```python
X = df.iloc[:,1:179].values
X.shape
```




    (11500, 178)




```python
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.2)
```


```python
from sklearn.metrics import accuracy_score, classification_report
```


```python
def evaluate(name, predictions, ytest):
    print(f'Accuracy of {name} : {accuracy_score(predictions, ytest) * 100} %')
    print(classification_report(predictions, ytest, target_names = ['Non Seizure', 'Seizure']))
```


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
#1. Logistic Regression
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(xtrain, ytrain)
predictions = LR.predict(xtest)
evaluate(LR, predictions, ytest)
```

    Accuracy of LogisticRegression() : 66.52173913043478 %
                  precision    recall  f1-score   support

     Non Seizure       0.73      0.83      0.77      1597
         Seizure       0.43      0.30      0.35       703

        accuracy                           0.67      2300
       macro avg       0.58      0.56      0.56      2300
    weighted avg       0.64      0.67      0.65      2300




```python
from sklearn.svm import SVC
SV = SVC(kernel = 'rbf', C = 2)
SV.fit(xtrain, ytrain)
predictions = SV.predict(xtest)
evaluate(SV, predictions, ytest)
```

    Accuracy of SVC(C=2) : 97.78260869565217 %
                  precision    recall  f1-score   support

     Non Seizure       1.00      0.98      0.99      1844
         Seizure       0.91      0.98      0.95       456

        accuracy                           0.98      2300
       macro avg       0.95      0.98      0.97      2300
    weighted avg       0.98      0.98      0.98      2300




```python
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = 16)
DT.fit(xtrain, ytrain)
predictions = DT.predict(xtest)
evaluate(DT, predictions, ytest)
```

    Accuracy of DecisionTreeClassifier(criterion='entropy', max_depth=16, random_state=0) : 94.26086956521739 %
                  precision    recall  f1-score   support

     Non Seizure       0.97      0.95      0.96      1847
         Seizure       0.83      0.89      0.86       453

        accuracy                           0.94      2300
       macro avg       0.90      0.92      0.91      2300
    weighted avg       0.94      0.94      0.94      2300




```python
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 20, random_state = 2, max_depth = 40)
RF.fit(xtrain, ytrain)
predictions = RF.predict(xtest)
evaluate(RF, predictions, ytest)
```

    Accuracy of RandomForestClassifier(max_depth=40, n_estimators=20, random_state=2) : 97.47826086956522 %
                  precision    recall  f1-score   support

     Non Seizure       0.98      0.98      0.98      1813
         Seizure       0.94      0.94      0.94       487

        accuracy                           0.97      2300
       macro avg       0.96      0.96      0.96      2300
    weighted avg       0.97      0.97      0.97      2300




```python
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(xtrain, ytrain)
predictions = NB.predict(xtest)
evaluate(NB, predictions, ytest)
```

    Accuracy of GaussianNB() : 95.78260869565217 %
                  precision    recall  f1-score   support

     Non Seizure       0.98      0.97      0.97      1828
         Seizure       0.88      0.92      0.90       472

        accuracy                           0.96      2300
       macro avg       0.93      0.94      0.94      2300
    weighted avg       0.96      0.96      0.96      2300




```python
from sklearn.neighbors import KNeighborsClassifier
KNN= KNeighborsClassifier(n_neighbors = 10)
KNN.fit(xtrain, ytrain)
predictions = KNN.predict(xtest)
evaluate(KNN, predictions, ytest)
```

    Accuracy of KNeighborsClassifier(n_neighbors=10) : 90.34782608695652 %
                  precision    recall  f1-score   support

     Non Seizure       1.00      0.89      0.94      2031
         Seizure       0.55      1.00      0.71       269

        accuracy                           0.90      2300
       macro avg       0.77      0.94      0.82      2300
    weighted avg       0.95      0.90      0.91      2300




```python
from xgboost import XGBClassifier
XGB = XGBClassifier(learning_rate = 0.01, n_estimators = 25,
                    max_depth = 25, gamma = 0.6,
                    subsample = 0.52, colsample_bytree = 0.6,
                    seed = 27, reg_lambda = 2, booster = 'dart',
                    colsample_bylevel = 0.6, colsample_bynode = 0.5
                   )
XGB.fit(xtrain, ytrain)
predictions = XGB.predict(xtest)
evaluate(XGB, predictions, ytest)
```

    Accuracy of XGBClassifier(base_score=0.5, booster='dart', colsample_bylevel=0.6,
                  colsample_bynode=0.5, colsample_bytree=0.6, gamma=0.6, gpu_id=-1,
                  importance_type='gain', interaction_constraints='',
                  learning_rate=0.01, max_delta_step=0, max_depth=25,
                  min_child_weight=1, missing=nan, monotone_constraints='()',
                  n_estimators=25, n_jobs=0, num_parallel_tree=1, random_state=27,
                  reg_alpha=0, reg_lambda=2, scale_pos_weight=1, seed=27,
                  subsample=0.52, tree_method='exact', validate_parameters=1,
                  verbosity=None) : 96.86956521739131 %
                  precision    recall  f1-score   support

     Non Seizure       0.99      0.97      0.98      1841
         Seizure       0.90      0.95      0.92       459

        accuracy                           0.97      2300
       macro avg       0.94      0.96      0.95      2300
    weighted avg       0.97      0.97      0.97      2300




```python
from mlxtend.classifier import StackingCVClassifier
SCV=StackingCVClassifier(classifiers=[XGB,SV,KNN, NB, RF, DT],meta_classifier= SV,random_state=42)
SCV.fit(xtrain, ytrain)
predictions = SCV.predict(xtest)
evaluate(SCV, predictions, ytest)
```


```python

```
