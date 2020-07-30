# import the python packages needed to generate simulated data for the tutorial
import os
import shutil
import scipy
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
# from matplotlib import pyplot
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ZscoreSustain import ZscoreSustain
from statsmodels.graphics.gofplots import qqplot
import sklearn.model_selection
from sklearn.covariance import EmpiricalCovariance
from sklearn import linear_model
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_curve, auc, confusion_matrix, balanced_accuracy_score, multilabel_confusion_matrix, classification_report # decent video guide: https://www.youtube.com/watch?v=TtIjAiSojFE
import pylab

from kde_ebm.mixture_model import fit_all_kde_models, fit_all_gmm_models
from kde_ebm import plotting

# ------------------------ FUNCTIONS FOR USE IN PIPELINE ---------------------------------------------------------------

def nu_freq(X_reg, y_reg): # Get regression params for each biomarker
    lr = linear_model.LinearRegression()
    lr.fit(X_reg, y_reg)
    return lr.coef_, lr.intercept_

def controls_residuals(X, y, covar, hdcat):
    controls_covar = covar[y == 0]
    controls_X = X[y == 0]
    n_biomarkers = controls_X.shape[1]
    regr_params = dict((x, None) for x in range(n_biomarkers)) # Creates an empty dict for each biomarker(containing (biomarker_num, None))
    for i in range(n_biomarkers):
        mask = ~np.isnan(controls_X[:, i]) & ~np.isnan(controls_covar).any(axis=1) # ???
        X_reg, y_reg = controls_covar[mask, :], controls_X[mask, i] # ???
        regr_params[i] = nu_freq(X_reg, y_reg) # Get regression coef and intercept for each biomarker
    residuals = X.copy() # Just creating a shallow copy of X
    for key, value in regr_params.items(): # TAKING THE COEFFICIENTS AWAY FROM EACH VALUE IN EACH BIOMARKER. For each biomarker in the regr_params dict
        residuals[:, key] -= regr_params[key][1] # resids[] = resids[] - regr_params[]
        residuals[:, key] -= np.matmul(covar, regr_params[key][0]) # resids[] = resids[] - regr_params[]
    print("residuals.shape(funct):", residuals.shape)
    print("hdcat.shape(funct):", hdcat.shape)
    residuals = np.append(residuals, hdcat[:, None], axis=1) # Putting hdcat back on for KDE / GMM models
    print("residuals.shape(funct2):", residuals.shape)
    return residuals # Produces adjusted residuals

# ---------------------------PIPELINE FUNCTION -------------------------------------------------------------------------

def pipeline(sustainType, df, return_stats=False):
    # (0) Subsetting data
    data_baseline_290 = df[df['visit'].values == ["Baseline"]].replace("<18", 17) # Selecting BASELINE visits only
    print("data_baseline_290(1):", data_baseline_290.shape)
    # print(data_baseline_290["age"].value_counts())
    print(data_baseline_290["hdcat"].value_counts())
    # data7 = data_baseline_290[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "age", "isced", "hdcat"]].dropna() # TEST - DELETE

    # Re-code hdcat values: 0=control, 1=manifest, 2=pre-manifest
    data_baseline_290["hdcat"].replace(3,1, inplace=True)
    data_baseline_290["hdcat"].replace(4,0, inplace=True)
    data_baseline_290["hdcat"].replace(5,0, inplace=True)

    print(data_baseline_290["hdcat"].value_counts())

    # Removing seemingly erroneous values - i.e., values outside the possible values in a scale
    data_baseline_290['isced'].value_counts() # Count values
    data_baseline_290 = data_baseline_290.drop(data_baseline_290[data_baseline_290.isced > 6].index) # isced is scored 0-6
    print("data_baseline_290(>6 dropped):", data_baseline_290.shape)

    # Splitting controls and disease so they can be labelled for control_residuals() - ALL PREDICTORS
    data_disease_290 = data_baseline_290[data_baseline_290['hdcat'].values == [1]]
    data_controls_290 = data_baseline_290[data_baseline_290['hdcat'].values == [0]]
    print("data_disease_290:", data_disease_290.shape)

    # Selecting predictors and covariates to control, disease and combined datasets
    data_disease_w_covariates = data_disease_290[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "age", "isced", "hdcat"]].dropna()  # *** NOTE: isced has some 13s in it (scale is 1-6)
    print("data_disease_w_covariates.shape:", data_disease_w_covariates.shape)
    # data_disease_w_covariates = data_disease_w_covariates.apply(pd.to_numeric)

    data_controls_w_covariates = data_controls_290[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "age", "isced", "hdcat"]].dropna()  # add "isced" sex and siteID

    # data_controls_w_covariates = data_controls_w_covariates.apply(pd.to_numeric) # Changing age from string to int
    data_combined_w_covariates = pd.concat([data_controls_w_covariates, data_disease_w_covariates])  # Needed for clinical labels
    data_combined_w_covariates = data_combined_w_covariates.apply(pd.to_numeric)  # Changing age from string to int
    print("data_combined_w_covariates:", data_combined_w_covariates)
    print("data_combined_w_covariates.shape:", data_combined_w_covariates.shape)
    # data_combined_w_covariates_w_covariates = pd.to_numeric(data_combined_w_covariates["age"]) # Changing age from string to int

    # # Selecting five predictors - control & disease COMBINED
    # data_contr_dis = data_combined_w_covariates[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore"]]

    # Selecting five predictors - control & disease SPLIT
    data_disease = pd.DataFrame(data_disease_w_covariates[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore"]])
    data_controls = pd.DataFrame(data_controls_w_covariates[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore"]])
    print("data_disease:", data_disease.shape)
    # N = data_disease.shape[1]

    # (1) Regressing out covariates
    # (1a) Getting input data for controls_residuals()
    array_of_covariate_data = np.array(data_combined_w_covariates[["age", "isced"]]) # # Selecting only the covariates

    # Adding labels to controls and disease sets  - then combining them
    data_disease.insert(0, 'disease/controls', 1)
    data_controls.insert(0, 'disease/controls', 0)
    data_combined_labelled = pd.concat([data_disease, data_controls])
    array_of_clinical_labels = np.array(data_combined_labelled.iloc[:, 0]) # Selecting only the label column
    array_of_data_to_be_adjusted = np.array(data_combined_labelled.iloc[:, 1:6]) # Selecting only the predictor columns

    hdcat = np.array(data_combined_w_covariates["hdcat"]) # selecting hdcat - it will be added back on at end of controls_residuals()
    # print("hdcat:", hdcat)
    print("hdcat.shape:", hdcat.shape)
    print("array_of_data_to_be_adjusted.shape", array_of_data_to_be_adjusted.shape)
    # swap in arrays for your data; shapes of each array as follows:
    # array_of_data_to_be_adjusted.shape = (Number of people, Number of biomarkers)
    # array_of_clinical_labels.shape = (Number of people)
    # array_of_covariate_data.shape = (Number of people, Number of covariates)

    # (1b) RUN controls residuals:
    data_adjusted = pd.DataFrame(controls_residuals(array_of_data_to_be_adjusted, array_of_clinical_labels, array_of_covariate_data, hdcat))
    print("data_adjusted:", data_adjusted.shape)
    data_adjusted.columns = ["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "hdcat"]

    if sustainType == 'mixture_GMM' or sustainType == "mixture_KDE":
        data = data_adjusted # Includes controls for KDE
        return data

    elif sustainType == "zscore":
        # Re-splitting updated data for step (2)
        array_of_clinical_labels = pd.DataFrame(array_of_clinical_labels)
        data_with_labels = pd.concat([data_adjusted, array_of_clinical_labels], axis=1)
        data_with_labels.columns = ["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "labels"]
        data = data_with_labels[data_with_labels['labels'].values == [1]].iloc[:, :5]
        data_controls = data_with_labels[data_with_labels['labels'].values == [0]].iloc[:, :5]

        # (2) Calculate the mean and standard deviation of each biomarker in your controls dataset
        mean_controls = np.mean(data_controls, axis=0)
        std_controls = np.std(data_controls, axis=0)

        # (3) Z-score your data by taking (data-mean_controls)/std_controls.
        # data_zscore = scipy.stats.zscore(data) #  Gets different results...
        data = (data - mean_controls) / std_controls
        print("data:", data.shape)
        data_controls = (data_controls - mean_controls) / std_controls

        ## (4) Identify any biomarkers that decrease with disease progression, these will have mean_data < mean_controls.
        # Multiply the data for these biomarkers by -1.
        # *** NEEDS AUTOMATED
        IS_decreasing_1 = np.mean(data, axis=0) < np.mean(data_controls, axis=0)
        print(IS_decreasing_1)
        data = pd.DataFrame((data - mean_controls) / std_controls)
        data_controls = pd.DataFrame((data_controls - mean_controls) / std_controls)
        data.iloc[:, 0] = data.iloc[:, 0].mul(-1)
        data.iloc[:, 2] = data.iloc[:, 2].mul(-1)
        data.iloc[:, 3] = data.iloc[:, 3].mul(-1)
        data.iloc[:, 4] = data.iloc[:, 4].mul(-1)
        IS_decreasing_2 = np.mean(data, axis=0) < np.mean(data_controls, axis=0)
        print(IS_decreasing_2)
        if return_stats==False:
            return data
        elif return_stats==True:
            # Check that the mean of the controls population is 0 (means covariate regression worked)
            print("\n", 'Mean of disease dataset is','\n', np.mean(data,axis=0))
            # Check that the standard deviation of the whole dataset is greater than 1
            print("\n", 'Standard deviation of disease dataset is','\n', np.std(data,axis=0))
            print("\n", 'Mean of controls is','\n', np.mean(data_controls,axis=0))
            # Check that the standard deviation of the controls population is 1
            print("\n", 'Standard deviation of controls is',' \n', np.std(data_controls,axis=0))
            # Check that the mean of the whole dataset is positive
            # print(IS_decreasing_1)
            # print(IS_decreasing_2)
            return data



