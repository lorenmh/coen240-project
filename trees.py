import numpy as np
import pandas as pd

from collections import defaultdict

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import sklearn

# mushroom data
m_le = defaultdict(LabelEncoder)
m_ohe = OneHotEncoder()

d = defaultdict(LabelEncoder)
m_data = pd.read_csv('datasets/agaricus-lepiota.data', header=None)
m_data = m_data.apply(lambda d: m_le[d.name].fit_transform(d))

m_X = m_data.values[:,1:]
m_X = m_ohe.fit_transform(m_X)
m_Y = m_data.values[:,0]

for i in range(10):
    m_X_train, m_X_test, m_Y_train, m_Y_test = train_test_split(
            m_X, m_Y, test_size=0.25, random_state=i
    )

    m_entropy = DecisionTreeClassifier(criterion='entropy')
    m_entropy.fit(m_X_train, m_Y_train)
    m_entropy_predict = m_entropy.predict(m_X_test)
    m_entropy_acc = accuracy_score(m_entropy_predict, m_Y_test)


    m_gini = DecisionTreeClassifier(criterion='gini')
    m_gini.fit(m_X_train, m_Y_train)
    m_gini_predict = m_entropy.predict(m_X_test)
    m_gini_acc = accuracy_score(m_gini_predict, m_Y_test)

    print(
        'iteration: %i, m_entropy_acc: %f, m_gini_acc: %f' % (
            i, m_entropy_acc, m_gini_acc
        )
    )

# iris data
i_data = pd.read_csv('datasets/iris.data', header=None)

i_X = i_data.values[:,:4]
i_Y = i_data.values[:,4]

for i in range(10):
    i_X_train, i_X_test, i_Y_train, i_Y_test = train_test_split(
            i_X, i_Y, test_size=0.25, random_state=i
    )

    i_entropy = DecisionTreeClassifier(criterion='entropy')
    i_entropy.fit(i_X_train, i_Y_train)
    i_entropy_predict = i_entropy.predict(i_X_test)
    i_entropy_acc = accuracy_score(i_entropy_predict, i_Y_test)

    i_gini = DecisionTreeClassifier(criterion='gini')
    i_gini.fit(i_X_train, i_Y_train)
    i_gini_predict = i_entropy.predict(i_X_test)
    i_gini_acc = accuracy_score(i_gini_predict, i_Y_test)

    print(
        'iteration: %i, i_entropy_acc: %f, i_gini_acc: %f' % (
            i, i_entropy_acc, i_gini_acc
        )
    )
