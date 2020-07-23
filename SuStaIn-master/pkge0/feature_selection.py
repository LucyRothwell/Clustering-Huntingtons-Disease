# FEATURE SELECTION

data_col6_onwards = data_baseline_290.iloc[:,6:len(data_baseline_290)]


def rf_feature_selection(data):
    # code here


# Encoding categorical data
# We have 1 categroical variable - hdcat
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_hdcat = LabelEncoder() # Creating object of LabelEncoder class for geog
data_col6_onwards[:, 1] = labelencoder_hdcat.fit_transform(data_col6_onwards[:, 1]) #Fitting object to the 2nd
    # variable in X, geogrpahy. Putting X[:, 1] before the = transforms the
    # geog column within X, as we are applying the object "labelencoder_X_1" to it.

onehotencoder = OneHotEncoder(categorical_features = [1]) # Tranforming to
    # dummy variables. Need to do because codes are not ordinal. Ex, France = 0 and
    # Spain = 2, but Spain is not higher in value. Doesnt need done for male/female
    # as, bc there is only two values (m/f) we can't remove one to avoid falling
    # into "dummy variable trap" (?)
data_col6_onwards = onehotencoder.fit_transform(data_col6_onwards).toarray() #Putting the variable before the =
    # again transforms it, this time using the onehotencoder object we created

data_col6_onwards = data_col6_onwards[:, 1:] # Removing one dummy coded column to avoid falling into "Dummy
    # variable trap". Ie taking all columns but the first one (0)
    # NB: Bc Y variables is already coded (0/1), we don't need to encode or transform

# Replece NaN with mean of column
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean', missing_values="nan")  #for mode we specify strategy='most_frequent'
data_columns = list(data_col6_onwards) #fancy imputation function removes column names, so we retain them and reassign them to the imputed data
imp.fit(data_col6_onwards)

# Split train, test set
first75 = round(len(data_col6_onwards)*0.75)
training_set = data_col6_onwards[0:first75]
test_set = data_col6_onwards[first75:len(data_col6_onwards)]

x_train = training_set.loc[:, training_set.columns != 'hdcat']
x_test = test_set.loc[:, test_set.columns != 'hdcat']

y_train = training_set[['hdcat']]
y_test = test_set[['hdcat']]

# RANDOM FOREST CLASSIFIER
rfmodel = RandomForestClassifier(n_estimators=24) #rfmodel is the function in sklearn
rfmodel.fit(x_train,y_train)
rfpredictions = rfmodel.predict(x_test)
resultRfmodel = accuracy_score(y_test, rfpredictions)
print('Random Forest accuracy: ', resultRfmodel)

# RANDOM FOREST IMPORTANCES
feature_list = list(x_train.columns)
# Get numerical feature importances
importances = list(rfmodel.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
importances = [('{:20} Importance: {}'.format(*pair)) for pair in feature_importances];
print(importances)
# ----------------------------------------------------------------------------------------------------------------------
