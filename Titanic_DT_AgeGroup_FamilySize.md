
# Titanic : An Analysis using Decsion Tree ( With Family )

Two new datasets named **titanic_train** and **titanic_test** have been created using 2 respective csvs i.e. **titanic_train.csv** and **titanic_test.csv**


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import tree
from sklearn import metrics
import matplotlib.pyplot as plt
%matplotlib inline

titanic_train = pd.read_csv('E:/CSVFiles/Titanic/train.csv')
titanic_test = pd.read_csv('E:/CSVFiles/Titanic/test.csv')
```


```python
titanic_train.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>

Calculating the **percentage of null values** in the **missing_data** dataframe. 

```python
total = titanic_train.isnull().sum().sort_values(ascending=False)
percent = (titanic_train.isnull().sum()/titanic_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cabin</th>
      <td>687</td>
      <td>0.771044</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>177</td>
      <td>0.198653</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>2</td>
      <td>0.002245</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ticket</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Name</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Survived</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>PassengerId</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>


It is seen that number of **Null** Values are **177** for **Age** and **687** for **Cabin** column 

```python
titanic_train.isnull().sum()

    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64
```



```python
titanic_train.columns[pd.isnull(titanic_train).sum() > 0].tolist()
    ['Age', 'Cabin', 'Embarked']
```

Creating a new column **Family_Size** for the data set **titanic_train**

```python
titanic_train['Family_Size'] = titanic_train['SibSp'] + titanic_train['Parch'] + 1
```
Dropping the corresponding columns **Name**, **Cabin**, and **Ticket** from the dataframe **titanic_train**

```
titanic_train.drop('Name',axis=1,inplace=True)
titanic_train.drop('Cabin',axis=1,inplace=True)
titanic_train.drop('Ticket',axis=1,inplace=True)
```
The mean value of age is 29.69 years. So we will plan to replace those **NaN** values against column **Age** with **29**

```python
titanic_train['Age'].mean()
    29.69911764705882
```


```python
titanic_train['Age'].fillna(value=29, inplace=True)
```


```python
titanic_train.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
      <th>Family_Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


A new function has been created called **Age_To_Groups** which will segragate the values of age into 5 main categories i.e.

**1) Age < 12 --> Child**
**2) 12 < Age < 19 --> Tenager**
**3) 19 < Age < 30 --> Youth**
**4) 30 < Age < 50 --> Middle_Aged**
**5) 50 < Age < 100 --> Senior_Citizen**


```python
def Age_To_Groups(age):

    if age < 12:
        return 'Child'
    elif 12 < age < 19:
        return 'Tenager'
    elif 19 < age < 30:
        return 'Youth'
    elif 30 < age < 50:
        return 'Middle_Aged'
    elif 50 < age < 100:
        return 'Senior_Citizen'
```

The function **Age_To_Groups** has been applied to the **Age** column in order to convert them into categorical values.

```python
titanic_train['Age_Group'] = titanic_train['Age'].apply(Age_To_Groups)
```

```python
titanic_train['Age_Group'].value_counts()

    Youth             397
    Middle_Aged       231
    Tenager            70
    Child              68
    Senior_Citizen     64
    Name: Age_Group, dtype: int64
```
After this all the categorical values been converted into dummy variables using **get_dummies** of titanic_train dataset

```python
titanic_train = pd.get_dummies(titanic_train)
```
Below is the dataset after the conversion

```python
titanic_train.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Family_Size</th>
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
      <th>Age_Group_Child</th>
      <th>Age_Group_Middle_Aged</th>
      <th>Age_Group_Senior_Citizen</th>
      <th>Age_Group_Tenager</th>
      <th>Age_Group_Youth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


Also we can see that there is no **NaN** values in the training dataset.

```python
titanic_train.isnull().sum()

    PassengerId                 0
    Survived                    0
    Pclass                      0
    Age                         0
    SibSp                       0
    Parch                       0
    Fare                        0
    Family_Size                 0
    Sex_female                  0
    Sex_male                    0
    Embarked_C                  0
    Embarked_Q                  0
    Embarked_S                  0
    Age_Group_Child             0
    Age_Group_Middle_Aged       0
    Age_Group_Senior_Citizen    0
    Age_Group_Tenager           0
    Age_Group_Youth             0
    dtype: int64
```

Dataset has been converted into X and Y arrays of features.

```python
Y = targets = labels = titanic_train['Survived'].values

columns = ["Fare","Pclass","Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S", "SibSp", "Parch","Family_Size","Age_Group_Child","Age_Group_Middle_Aged","Age_Group_Senior_Citizen","Age_Group_Tenager","Age_Group_Youth"]
features = titanic_train[list(columns)].values
features
    array([[ 7.25  ,  3.    ,  0.    , ...,  0.    ,  0.    ,  1.    ],
           [71.2833,  1.    ,  1.    , ...,  0.    ,  0.    ,  0.    ],
           [ 7.925 ,  3.    ,  1.    , ...,  0.    ,  0.    ,  1.    ],
           ...,
           [23.45  ,  3.    ,  1.    , ...,  0.    ,  0.    ,  1.    ],
           [30.    ,  1.    ,  0.    , ...,  0.    ,  0.    ,  1.    ],
           [ 7.75  ,  3.    ,  0.    , ...,  0.    ,  0.    ,  0.    ]])

```

```python
X = features
X
    array([[ 7.25  ,  3.    ,  0.    , ...,  0.    ,  0.    ,  1.    ],
           [71.2833,  1.    ,  1.    , ...,  0.    ,  0.    ,  0.    ],
           [ 7.925 ,  3.    ,  1.    , ...,  0.    ,  0.    ,  1.    ],
           ...,
           [23.45  ,  3.    ,  1.    , ...,  0.    ,  0.    ,  1.    ],
           [30.    ,  1.    ,  0.    , ...,  0.    ,  0.    ,  1.    ],
           [ 7.75  ,  3.    ,  0.    , ...,  0.    ,  0.    ,  0.    ]])

```


```python
my_tree_one = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
my_tree_one = my_tree_one.fit(X,Y)
my_tree_one

    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')
```

```python
print(my_tree_one.feature_importances_)
print(my_tree_one.score(X,Y))
list(zip(columns,my_tree_one.feature_importances_))

    [0.07618531 0.1868736  0.         0.5677067  0.         0.
     0.         0.         0.         0.08384035 0.08539404 0.
     0.         0.         0.        ]
    0.8215488215488216

    [('Fare', 0.07618531188617528),
     ('Pclass', 0.18687359962400962),
     ('Sex_female', 0.0),
     ('Sex_male', 0.5677067003586163),
     ('Embarked_C', 0.0),
     ('Embarked_Q', 0.0),
     ('Embarked_S', 0.0),
     ('SibSp', 0.0),
     ('Parch', 0.0),
     ('Family_Size', 0.08384034573912937),
     ('Age_Group_Child', 0.08539404239206941),
     ('Age_Group_Middle_Aged', 0.0),
     ('Age_Group_Senior_Citizen', 0.0),
     ('Age_Group_Tenager', 0.0),
     ('Age_Group_Youth', 0.0)]
```



```python
# Since Fare and Pclass are both related to each other and the feature importance of Pclass is more than Fare,
#hence Fare is removed
columns = ["Pclass","Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S", "SibSp", "Parch","Family_Size","Age_Group_Child","Age_Group_Middle_Aged","Age_Group_Senior_Citizen","Age_Group_Tenager","Age_Group_Youth"]
features = titanic_train[list(columns)].values
features
```




    array([[3, 0, 1, ..., 0, 0, 1],
           [1, 1, 0, ..., 0, 0, 0],
           [3, 1, 0, ..., 0, 0, 1],
           ...,
           [3, 1, 0, ..., 0, 0, 1],
           [1, 0, 1, ..., 0, 0, 1],
           [3, 0, 1, ..., 0, 0, 0]], dtype=int64)




```python
X_nofare = features
X_nofare
```




    array([[3, 0, 1, ..., 0, 0, 1],
           [1, 1, 0, ..., 0, 0, 0],
           [3, 1, 0, ..., 0, 0, 1],
           ...,
           [3, 1, 0, ..., 0, 0, 1],
           [1, 0, 1, ..., 0, 0, 1],
           [3, 0, 1, ..., 0, 0, 0]], dtype=int64)




```python
my_tree_one_nofare = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
my_tree_one_nofare = my_tree_one.fit(X_nofare,Y)
my_tree_one_nofare
```




    DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=3,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




```python
print(my_tree_one_nofare.feature_importances_)
print(my_tree_one_nofare.score(X_nofare,Y))
list(zip(columns,my_tree_one_nofare.feature_importances_))
```

    [0.26338546 0.60250474 0.         0.         0.         0.00523258
     0.         0.         0.0501605  0.07871672 0.         0.
     0.         0.        ]
    0.8148148148148148
    




    [('Pclass', 0.2633854609255124),
     ('Sex_female', 0.6025047440218411),
     ('Sex_male', 0.0),
     ('Embarked_C', 0.0),
     ('Embarked_Q', 0.0),
     ('Embarked_S', 0.005232575039177103),
     ('SibSp', 0.0),
     ('Parch', 0.0),
     ('Family_Size', 0.05016050410292612),
     ('Age_Group_Child', 0.07871671591054341),
     ('Age_Group_Middle_Aged', 0.0),
     ('Age_Group_Senior_Citizen', 0.0),
     ('Age_Group_Tenager', 0.0),
     ('Age_Group_Youth', 0.0)]




```python
with open("titanic_1_2.dot", 'w') as f:
    f = tree.export_graphviz(my_tree_one, out_file=f, feature_names=columns)
```


```python
#### TEST Dataset #####

titanic_test.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
total = titanic_test.isnull().sum().sort_values(ascending=False)
percent = (titanic_test.isnull().sum()/titanic_test.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data_test.head(20)
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
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Cabin</th>
      <td>327</td>
      <td>0.782297</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>86</td>
      <td>0.205742</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>1</td>
      <td>0.002392</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Ticket</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Name</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>PassengerId</th>
      <td>0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_test.drop('Cabin',axis=1,inplace=True)
titanic_test_name=titanic_test['Name']
titanic_test.drop('Name',axis=1,inplace=True)
titanic_test.drop('Ticket',axis=1,inplace=True)
```


```python
titanic_test.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
titanic_test = pd.get_dummies(titanic_test)
```


```python
titanic_test.head()
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
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Sex_female</th>
      <th>Sex_male</th>
      <th>Embarked_C</th>
      <th>Embarked_Q</th>
      <th>Embarked_S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
columns = ["Pclass","Sex_female","Sex_male","Embarked_C","Embarked_Q","Embarked_S", "Age", "SibSp", "Parch"]
features = titanic_test[list(columns)].values
features

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
X_test = imp.fit_transform(features)
X_test
```




    array([[ 3.        ,  0.        ,  1.        , ..., 34.5       ,
             0.        ,  0.        ],
           [ 3.        ,  1.        ,  0.        , ..., 47.        ,
             1.        ,  0.        ],
           [ 2.        ,  0.        ,  1.        , ..., 62.        ,
             0.        ,  0.        ],
           ...,
           [ 3.        ,  0.        ,  1.        , ..., 38.5       ,
             0.        ,  0.        ],
           [ 3.        ,  0.        ,  1.        , ..., 30.27259036,
             0.        ,  0.        ],
           [ 3.        ,  0.        ,  1.        , ..., 30.27259036,
             1.        ,  1.        ]])




```python
pred = my_tree_one.predict(X_test)
pred
```




    array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1,
           1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
           1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
           1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
           1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
           0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
           1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
           0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
           1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,
           0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
           1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1,
           0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
          dtype=int64)




```python
# Printing the Confusion Matrix

pred = my_tree_one.predict(X)
df_confusion = metrics.confusion_matrix(Y,pred)
df_confusion
```




    array([[511,  38],
           [122, 220]], dtype=int64)




```python
def plot_confusion_matrix(df_confusion, title='Confusion Matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
```


```python
 plot_confusion_matrix(df_confusion)
```


![png](output_37_0.png)



```python
# Changing the depth of the model

n = 15
s = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = n, min_samples_split=s, random_state=1)
my_tree_two = my_tree_two.fit(X,Y)

#Print the score of the new decision tree
print(my_tree_two.score(X,Y))
print(my_tree_one.score(X,Y))
```

    0.9068462401795735
    0.8204264870931538
    


```python
import numpy as np
#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

max_depth = np.arange(2,30,1)
score_dt_md = []

for n in max_depth:
    my_tree_two = tree.DecisionTreeClassifier(max_depth = n, min_samples_split=5, random_state=1)
    my_tree_two = my_tree_two.fit(X,Y)
    score_dt_md.append(my_tree_two.score(X,Y))

plt.figure(figsize=(10,5))
plt.plot(max_depth,score_dt_md,color='b')
plt.xlabel('Max_Depth')
plt.ylabel('Score')
    
```




    Text(0,0.5,'Score')




![png](output_39_1.png)



```python
import numpy as np
#from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

min_samples_split = np.arange(2,30,1)
score_dt_sample_split = []

for n in min_samples_split:
    my_tree_two = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split=n, random_state=1)
    my_tree_two = my_tree_two.fit(X,Y)
    score_dt_sample_split.append(my_tree_two.score(X,Y))

plt.figure(figsize=(10,5))
plt.plot(min_samples_split,score_dt_sample_split,color='r')
plt.xlabel('min_samples_split when max_depth')
plt.ylabel('Score')
    
```




    Text(0,0.5,'Score')




![png](output_40_1.png)



```python
pred = my_tree_two.predict(X)

df_confusion = metrics.confusion_matrix(Y,pred)
df_confusion
```




    array([[502,  47],
           [ 88, 254]], dtype=int64)




```python
def plot_confusion_matrix(df_confusion, title='Confusion Matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

plot_confusion_matrix(df_confusion)
```


![png](output_42_0.png)



```python
# Try out family size
# RANDOM fOREST

from sklearn.ensemble import RandomForestClassifier

# Building and fitting my forest

forest = RandomForestClassifier(max_depth=15, min_samples_split=2, n_estimators=100, random_state=1)
my_forest = forest.fit(X,Y)

#Print the score of the fitted random forest
print(my_forest.score(X,Y))
```

    0.9371492704826038
    


```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

max_depth = np.arange(2,30,1)
scorerf = []

for n in max_depth:
    forest = RandomForestClassifier(max_depth=n, min_samples_split=5, n_estimators=100, random_state=1)
    my_forest = forest.fit(X,Y)
    scorerf.append(my_forest.score(X,Y))

plt.figure(figsize=(10,5))
plt.plot(max_depth,scorerf,color='b')
plt.xlabel('Max_Depth')
plt.ylabel('Score')
    
```




    Text(0,0.5,'Score')




![png](output_44_1.png)



```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

min_samples_split = np.arange(2,30,1)
score_rf_sample_split = []

for n in min_samples_split:
    forest = RandomForestClassifier(max_depth=10, min_samples_split=n, n_estimators=100, random_state=1)
    my_forest = forest.fit(X,Y)
    score_rf_sample_split.append(my_forest.score(X,Y))

plt.figure(figsize=(10,5))
plt.plot(min_samples_split,score_rf_sample_split,color='r')
plt.xlabel('min_samples_split when max_depth')
plt.ylabel('Score')
```




    Text(0,0.5,'Score')




![png](output_45_1.png)



```python
pred = my_forest.predict(X)
df_confusion = metrics.confusion_matrix(Y,pred)
df_confusion
```




    array([[515,  34],
           [103, 239]], dtype=int64)




```python
pred_test = my_forest.predict(X_test)
pred_test
```




    array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1,
           1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1,
           1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1,
           1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1,
           1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1,
           1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1,
           0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,
           1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
           0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
           0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
           0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
           1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0,
           0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0,
           1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
           0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
          dtype=int64)




```python
X = min_samples_split
Y = max_depth
Z = score_dt_sample_split
W = score_dt_md
```


```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

X = min_samples_split
Y = max_depth
Z = score_dt_sample_split

ax.scatter(X, Y, Z, c='r', marker='*')
ax.set_xlabel('min_samples_split axis')
ax.set_ylabel('max_depth axis')
ax.set_zlabel('score_dt_sample_split axis')

plt.show()
```


![png](output_49_0.png)



```python
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111, projection='3d')

X = min_samples_split
Y = max_depth
W = score_dt_md

ax.scatter(X, Y, W, c='r', marker='*')
ax.set_xlabel('min_samples_split axis')
ax.set_ylabel('max_depth axis')
ax.set_zlabel('score_dt_md axis')

plt.show()
```


![png](output_50_0.png)

