# FEATURE SELECTION

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import scipy.stats
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import collections
import operator
import inspect

#######################################################################################################################

# CONTENTS
# (1) Hypothesis test feature selection
# (2) Random forest non-parametric feature selection

#######################################################################################################################


# --------------------------- (1) HYPOTHESIS TEST FEATURE SELECTION -------------------------------------------------------------
# 1. Find mean and SD of each biomarker
# 2. Do tests between disease and controls
# 3. Select features with biggest effect size (difference between groups)

def feature_selection(data_all_features, target, method="t_test", sign_lev=0.05): # Data and controls
    results = {} # Create dictionary to store p-values of each biomarker
    count = 0
    for column in data_all_features:
        controls = data_all_features [data_all_features[target] == 0]
        disease = data_all_features [data_all_features[target] == 1]
        if method=="t_test":
            score = scipy.stats.ttest_ind(disease[column], controls[column], nan_policy='omit')
        elif method=="wilc":
            score = scipy.stats.wilcoxon(x=disease[column], y=controls[column])
        elif method=="mann_w":
            score = scipy.stats.mannwhitneyu(disease[column], controls[column])
        else:
            print("Error: method must be either 't_test', 'wilc' or 'mann_w'")
        if score[1] < sign_lev:
            # count = count + 1
            results[column] = [round(abs(score[0]),2), round(score [1],4)] # adding key and value to dict
    del results[target]
    results = collections.OrderedDict(sorted(results.items(), key=operator.itemgetter(1), reverse=True)) # Sorting dict by t-stat (changes dict to list) REFERENCE: https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    print("\n","---", method, "STATS WITH P <", sign_lev, "---")
    for i in results:
        print("     {:10}    Stat: {:5}     P-value: {:5}".format(i, results[i][0], results[i][1]))

    ## Print results to CV instead
    # feature_ranks = open("feature_ranks.csv", 'a')
    # feature_ranks.write("feature[wilc-stat, p-val]" + '\n' + '\n')
    # for i in results.keys():
    #     feature_ranks.write(i + str(results[i]) + '\n')
    # feature_ranks.close()

    return results



# --------------------------- (2) RAND. FOREST FEATURE SELECTION -----------------------------------------------------------

def feature_selection_rf(data, target):
    # data = data.iloc [:, 7:] # Ignoring first 7 features (subjectID etc)
    data = data [data["hdcat"] != 2] # Removing pre-manifest so have only manifest disease and controls
    data.dropna(axis=1, how='all', inplace=True)
    data.fillna(data.mean(), inplace=True)

    X = data.loc[:, data.columns != target] # All except target
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=18)

    # RANDOM FOREST CLASSIFIER
    rfmodel = RandomForestClassifier(n_estimators=24)
    rfmodel.fit(x_train,y_train)
    rfpredictions = rfmodel.predict(x_test)
    resultRfmodel = accuracy_score(y_test, rfpredictions)
    print('Random Forest accuracy: ', resultRfmodel)

    # RANDOM FOREST IMPORTANCES
    feature_list = list(x_train.columns)
    # Get numerical feature importances
    importances = list(rfmodel.feature_importances_)
    # List of tuples with variable and importance
    feature_importances =  [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x [1], reverse = True)
    # Print out the feature and importances
    [print('{:20} Importance: {}'.format(*pair)) for pair in feature_importances];

    code = inspect.getsource(RandomForestClassifier)





