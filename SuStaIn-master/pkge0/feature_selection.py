# FEATURE SELECTION

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import scipy.stats
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
import numpy as np
import pandas as pd

#######################################################################################################################

# CONTENTS
# (1) Wilcoxon non-parametric feature selection
# (2) Random forest non-parametric feature selection
# (3) T-test parametric feature selection

#######################################################################################################################


# --------------------------- (1) WILCOXON FEATURE SELECTION ------------------------------------------------------------------


def feature_selection_wilcoxon(data_all_features, sign_lev = 0.05):
    data_all_features = data_all_features.iloc[:, 1:] # Skip subjid
    wilc_results = {} # Create dictionary to store p-values of each biomarker
    count = 0
    for column in data_all_features:
        controls = data_all_features[data_all_features["hdcat"] == 0]
        disease = data_all_features[data_all_features["hdcat"] == 1].iloc[:controls.shape[0], :] # making disease same length as controls (wilcoxon requires it)
        wilc_score = scipy.stats.wilcoxon(x=disease[column], y=controls[column])
        if wilc_score[1] < sign_lev:
            count = count + 1
            wilc_results[column] = [round(wilc_score[0],2), round(wilc_score[1],4)] # adding key and value to dict
    wilc_results = sorted(wilc_results.items(), key=operator.itemgetter(1)) # Sorting dict by t-stat (changes dict to list)
    wilc_results = collections.OrderedDict(wilc_results) # Converting list back to dict
    wilc_csv = open("wilc.csv", 'a')
    wilc_csv.write("feature[wilc-stat, p-val]" + '\n' + '\n')
    for i in wilc_results.keys():
        wilc_csv.write(i + str(wilc_results[i]) + '\n')
    wilc_csv.close()
    return wilc_results



# --------------------------- (2) RAND. FOREST FEATURE SELECTION -----------------------------------------------------------

def feature_selection_rf(data):
    # data = data.iloc[:, 7:] # Ignoring first 7 features (subjectID etc)
    data = data[data["hdcat"] != 2] # Removing pre-manifest so have only manifest disease and controls
    data.dropna(axis=1, how='all', inplace=True)
    data.fillna(data.mean(), inplace=True)

    # Split train, test set
    first75 = round(len(data)*0.75)
    training_set = data[0:first75]
    test_set = data[first75:len(data)]

    x_train = training_set.loc[:, training_set.columns != 'hdcat'] # Selecting all cols except hdcat
    y_train = training_set["hdcat"].round()

    x_test = test_set.loc[:, test_set.columns != 'hdcat']
    y_test = test_set["hdcat"].round()

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
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    # importances = [('{:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    # print(importances)



# --------------------------- (3) T-TEST FEATURE SELECTION ------------------------------------------------------------------
# 1. Find mean and SD of each biomarker
# 2. Do T - tests between disease and controls
# 3. Select features with biggest effect size (difference between groups)
import collections
import operator
def feature_selection_ttest(data_all_features, sign_lev=0.05): # Data and controls
    data_all_features = data_all_features.iloc[:, 1:]  # Skip subjid
    ttest_results = {} # Create dictionary to store p-values of each biomarker
    count = 0
    for column in data_all_features:
        controls = data_all_features[data_all_features["hdcat"] == 0]
        disease = data_all_features[data_all_features["hdcat"] == 1]
        t_test_score = scipy.stats.ttest_ind(disease[column], controls[column], nan_policy='omit')
        if t_test_score[1] < sign_lev:
            count = count + 1
            ttest_results[column] = [round(t_test_score[0],2), round(t_test_score[1],4)] # adding key and value to dict
    ttest_results = sorted(ttest_results.items(), key=operator.itemgetter(1)) # Sorting dict by t-stat (changes dict to list)
    ttest_results = collections.OrderedDict(ttest_results) # Converting list back to dict
    ttest_csv = open("ttest_results.csv", 'a')
    ttest_csv.write("feature[t-stat, p-val]" + '\n' + '\n')
    for i in ttest_results.keys():
        ttest_csv.write(i + str(ttest_results[i]) + '\n')
    ttest_csv.close()
    return ttest_results

