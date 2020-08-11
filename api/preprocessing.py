import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

le = LabelEncoder()
ss = StandardScaler()
mms = MinMaxScaler(feature_range=(0, 1))

def comp_classification(df, dv):
    for i in df.columns:
        df[i].replace('-', np.nan)
        df[i].replace('+', np.nan)
        if (i[-1] == 'D' or i[-1] == 'd') and (i[-2] == 'I' or i[-2] == 'i'):
            df = df.drop([i], axis=1)

    for i in df.columns:
        if df[i].isnull().sum() > 0:
            if df[i].dtype == 'object':
                df = df.dropna(subset=[i])
            else:
                df[i] = 0

    for i in df.columns:
        if df[i].nunique() > 0 and df[i].dtype != 'int64' and df[i].dtype != 'float64' and df[i].dtype != 'int32':
           df[i] = le.fit_transform(df[i])

    y = df.iloc[:, dv]
    x = df.iloc[:, 0:dv]

    '''Splitting dataset into training and testing set'''

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    '''feature scaling'''
    # fit the object on training set and then transform

    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    # ----------------------------------------------------------------------------------------------

    accuracy_dict = {}
    max_score = 0
    max_models = []

    models = {
              'LogisticRegression_Classifier': LogisticRegression(),
              'NaiveBayes_Classifier': GaussianNB(),
              'KNN_Classifier': KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski'),
              'SVM_Classifier': SVC(kernel='linear'),
              'KernalSVM_Classifier': SVC(kernel='rbf'),
              'DecisionTree_Classifier': DecisionTreeClassifier(criterion="entropy"),
              'RandomForest_Classifier': RandomForestClassifier(n_estimators=10, criterion='entropy')
    }
    for key, value in models.items():
        model = value
        model.fit(x_train, y_train)
        accuracy_dict.update({key: round(model.score(x_test, y_test), 3) * 100})
        if accuracy_dict[key] > max_score:
            max_score = accuracy_dict[key]
            max_models.clear()
            max_models.append(key)
        elif accuracy_dict[key] == max_score:
            max_models.append(key)
        else:
            continue

    # returning the accuracies
    # return {'MA':models_accuracies,'max_accuracy':max_accuracy,'maxx_models':maxx_models }
    return {'MA': accuracy_dict, 'max_accuracy': max_score, 'maxx_models': max_models}


def comp_regression(df, dv):
    for i in df.columns:
        df[i].replace('-', np.nan)
        df[i].replace('+', np.nan)
        if (i[-1] == 'D' or i[-1] == 'd') and (i[-2] == 'I' or i[-2] == 'i'):
            df = df.drop([i], axis=1)

    for i in df.columns:
        if df[i].isnull().sum() > 0:
            if df[i].dtype == 'object':
                df = df.dropna(subset=[i])
            else:
                df[i] = 0

    for i in df.columns:
        if df[i].nunique() > 0 and df[i].dtype != 'int64' and df[i].dtype != 'float64' and df[i].dtype != 'int32':
            df[i] = le.fit_transform(df[i])

    y = df.iloc[:, dv]
    x = df.iloc[:, 0:dv]

    '''Splitting dataset into training and testing set'''

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    '''feature scaling'''
    # fit the object on training set and then transform

    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)

    # ----------------------------------------------------------------------------------------------

    accuracy_dict = {}
    max_score = 0
    max_models = []
    # , 'SVR_Classifier': SVR(kernel='rbf'),
    #
    models = {'LinearRegression_Classifier': LinearRegression(),
              'PolynomialRegression': PolynomialFeatures(degree=2)}
    for key, value in models.items():
        model = value
        model.fit(x_train, y_train)
        accuracy_dict.update({key: round(model.score(x_test, y_test), 3) * 100})
        if accuracy_dict[key] > max_score:
            max_score = accuracy_dict[key]
            max_models.clear()
            max_models.append(key)
        elif accuracy_dict[key] == max_score:
            max_models.append(key)
        else:
            continue

    # returning the accuracies
    # return {'MA':models_accuracies,'max_accuracy':max_accuracy,'maxx_models':maxx_models }
    return {'MA': accuracy_dict, 'max_accuracy': max_score, 'maxx_models': max_models}

def comp_clustering(df, dv):
    pass