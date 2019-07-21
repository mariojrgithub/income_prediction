import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("./adult-income-dataset/adult.csv")

# filling missing values with mode value
col_names = df.columns
for c in col_names:
    df[c] = df[c].replace("?", np.NaN)

df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

# clean up column names
df.replace(['Divorced', 'Married-AF-spouse',
              'Married-civ-spouse', 'Married-spouse-absent',
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace = True)

#label Encoder
category_col =['workclass', 'race',
                'education','marital-status', 'occupation',
                             'relationship', 'gender', 'native-country', 'income']
labelEncoder = LabelEncoder()

# creating a map of all the numerical values of each categorical labels.
mapping_dict={}
for col in category_col:
    df[col] = labelEncoder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
    mapping_dict[col]=le_name_mapping
print(mapping_dict)

#droping redundant columns
df=df.drop(['fnlwgt','educational-num'], axis=1)

# set X and y variables
X = df.values[:, 0:12]
y = df.values[:,12]

# train test split
X_train, X_test, y_train, y_test = train_test_split( X, y,
                                                    test_size = 0.3,
                                                    random_state = 100)

# instantiate decision tree
dt_clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
                                    max_depth=5, min_samples_leaf=5)
# fit model to train data
dt_clf_gini.fit(X_train, y_train)
# make predictions
y_pred_gini = dt_clf_gini.predict(X_test)

# print accuracy score
print ("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test,y_pred_gini))

#creating and training a model
#serializing our model to a file called model.pkl
import pickle
pickle.dump(dt_clf_gini, open(".../model.pkl","wb"))                                                
