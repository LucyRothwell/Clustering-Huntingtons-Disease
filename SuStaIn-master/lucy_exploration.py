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
from scipy.stats import ttest_ind, norm
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_curve, auc, confusion_matrix, balanced_accuracy_score, multilabel_confusion_matrix, classification_report # decent video guide: https://www.youtube.com/watch?v=TtIjAiSojFE
import pylab

from kde_ebm.mixture_model import fit_all_kde_models, fit_all_gmm_models
from kde_ebm import plotting

# --------------LOADING/ SUBSETTING DATA--------------------------------------------------------------------------------
data = pd.read_csv("/Users/lucyrothwell/Google_Drive/MSc_Comp/9_Dissertation HD/Data - Enroll HD/enroll.csv", delimiter=',')
print("data (1):", data.shape)

# Random exploration
data["hdcat"].value_counts()#/len(data)*100
hdcat_4 = data[data["hdcat"].values == 4]
hdcat_5 = data[data["hdcat"].values == 5]

data["hdcat"].hist()
hdcat_4["hdcat"].hist()
hdcat_5["hdcat"].hist()


# (0) Subsetting sata
data_baseline_290 = data[data['visit'].values == ["Baseline"]] # Selecting BASELINE visits only
data_baseline_290 = data_baseline_290.replace("<18", 17) # Replace <18 with 17 in age column
print("data_baseline_290:", data_baseline_290.shape)

# Splitting controls and disease - ALL PREDICTORS
data_disease_290 = data_baseline_290[data_baseline_290['hdcat'].values != [4]]
data_controls_290 = data_baseline_290[data_baseline_290['hdcat'].values == [4]]
print("data_disease_290", data_disease_290.shape)

# Checking if 3 (gen_neg) and 4 (family_controls) have similar dists - if so, we can lump them together under controls

data_manifest_3 = data_baseline_290[data_baseline_290['hdcat'].values == [3]]
data_manifest_3 = data_manifest_3[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore"]]

data_gen_neg_4 = data_baseline_290[data_baseline_290['hdcat'].values == [4]]
data_gen_neg_4 = data_gen_neg_4[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore"]]

data_fam_control_5 = data_baseline_290[data_baseline_290['hdcat'].values == [5]]
data_fam_control_5 = data_fam_control_5[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore"]]


data_manifest_3.hist()
data_gen_neg_4.hist()
data_fam_control_5.hist()


# Adding covariates to control and disease SPLIT data
data_disease_w_covariates = data_disease_290[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "age", "isced"]].dropna() # *** NOTE: isced has some 13s in it (scale is 1-6)
print("data_disease_w_covariates:", data_disease_w_covariates.shape)
# data_disease_w_covariates = data_disease_w_covariates.apply(pd.to_numeric)
data_controls_w_covariates = data_controls_290[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "age", "isced"]].dropna() # add "isced" sex and siteID
# data_controls_w_covariates = data_controls_w_covariates.apply(pd.to_numeric) # Changing age from string to int
data_combined_w_covariates = pd.concat([data_controls_w_covariates, data_disease_w_covariates]) # Needed for clinical labels
data_combined_w_covariates = data_combined_w_covariates.apply(pd.to_numeric) # Changing age from string to int
# data_combined_w_covariates_w_covariates = pd.to_numeric(data_combined_w_covariates["age"]) # Changing age from string to int


# # Selecting five predictors - control & disease COMBINED
# data_contr_dis = data_combined_w_covariates[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore"]]

# Selecting five predictors - control & disease SPLIT
data_disease = pd.DataFrame(data_disease_w_covariates[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore"]])
data_controls = pd.DataFrame(data_controls_w_covariates[["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore"]])
print("data_disease:", data_disease.shape)


# ------------------EXPLORING DATA -------------------------------------------------------------------------------------
# % of each value per column
# data_disease["motscore"].value_counts()/len(data_disease)

## Normality check
# Histogram
data_disease.hist()
# KDE
data_disease["tfcscore"].plot.density() #KDE
# PDF.... https://www.kite.com/python/docs/scipy.stats.norm.pdf
x = data_disease["tfcscore"]
w = 4
h = 3
d = 70
plt.figure(figsize=(w, h), dpi=d)
x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
plt.plot(x, norm.pdf(x))
plt.savefig("out.png")

# # Normality test
# scipy.stats.shapiro(data_disease["motscore"])
# scipy.stats.anderson(data_disease["motscore"], dist='norm')

# FEATURES (All part of Baseline visit)
# ----- 0 often means perfect score
# motscore = UHDRS motor score (TMS) ------ 20%=0, 71%=1 (baseline disease). Score up to 124. The UHDRS‐TMS is formed of 15 items and has a maximum score of 124. The different items of the UHDRS‐TMS include chorea, dystonia, parkinsonism, motor performance, oculomotor function, and balance.
# tfcscore = UHDRS total functional capacity score ------ 26%=0, 34%=1  (baseline disease). UHDRS Total Functional Capacity (TFC) **KEY DIAGNOSTIC VAR
# mmsetotal = mini mental state examination (MMSE) ------ 24%=0, 20%=1 (baseline disease). MMSE is a 30-point test. https://www.ncbi.nlm.nih.gov/projects/gap/cgi-bin/GetPdf.cgi?id=phd001525.1
# irascore = irritability aggression ------ 31%=0, 17%=1 (baseline disease). Problem Behaviours Assessment – Short (PBA‐s). The short version of the Problem Behaviours Assessment (PBA-s) is a semi-structured interview containing 11 items, each designed to measure the severity and frequency of a different behavioural symptom in HD. http://eprints.whiterose.ac.uk/96137/1/15-164%20JHD%20revised%20manuscript%20-%20accepted%20version.pdf
# exfscore = executive function ------ 51%=0, 13%=1 (baseline disease). Problem Behaviours Assessment – Short (PBA‐s).

# diagconf - Diagnostic confidence level of motor abnormalities
# fascore - functional assessment score ------ 31%=0, 32%=1 (baseline disease)

# # User missing value counts
# len(data[data.exfscore == 9996])
# len(data[data.exfscore == 9997])
# len(data[data.exfscore == 9998])

# ---------SETTINGS-----------------------------------------------------------------------------------------------------
# N = data_disease.shape[1]        # number of biomarkers
# M = data_contr_dis.shape[0]       # number of observations ( e.g. subjects )
# M_controls = data_controls.shape[0]       # number of these that are controls subjects


# ----------PREPARE THE DATA FOR SUSTAIN --------------------------------------------------------------------------------

# (1) Regress out the effects of covariates. Learn the effects of covariates in a controls population and use this model
# to regress out the effect of covariates for all the subjects. Learning the model in the controls population will avoid
# regressing out disease effects, which you want to keep in your dataset.

# # (1a) Encoding categorical covariates: "siteID"
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_isced = LabelEncoder() # Creating object of LabelEncoder class for geog
# data_combined_w_covariates["isced"] = labelencoder_isced.fit_transform(data_combined_w_covariates["isced"]) #Fitting object to the 2nd variable in X, geogrpahy. Putting X[:, 1] before the = transforms the  geog column within X, as we are applying  the object "labelencoder_X_1" to it.
# # labelencoder_siteID = LabelEncoder() # Same for gender
# # data_baseline_290["siteID"] = labelencoder_siteID.fit_transform(data_baseline_290["siteID"])
# onehotencoder = OneHotEncoder(categories = "auto") # Transforming to
#     # dummy variables. Need to do because codes are not ordinal. Ex, France = 0 and
#     # Spain = 2, but Spain is not higher in value. Doesnt need done for male/female
#     # as, bc there is only two values (m/f) we can't remove one to avoid falling
#     # into "dummy variable trap" (?)
# data_combined_w_covariates = onehotencoder.fit_transform(data_combined_w_covariates).toarray() #Putting the variable before the =
#     # again transforms it, this time using the onehotencoder object we created
# data_combined_w_covariates = data_combined_w_covariates["isced"] # Removing one dummy coded column to avoid falling into "Dummy
#     # variable trap". Ie taking all columns but the first one (0)
#     # NB: Bc Y variables is already coded (0/1), we don't need to encode or transform
#
# # for i in data_combined_w_covariates["age"].values:
# #     # print(type(i))
# #     if type(i) == str:
# #         print(i)
# #
# # for i in data_combined_w_covariates["age"].values:
# #     # print(type(i))
# #     i = int(i)

# (1b) Covariate adjustment code

def nu_freq(X_reg, y_reg): # Get regression params for each biomarker
    lr = linear_model.LinearRegression()
    lr.fit(X_reg, y_reg)
    return lr.coef_, lr.intercept_

def controls_residuals(X, y, covar):
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
    return residuals # Produces adjusted residuals

# (1c) Getting input data for controls_residuals()
array_of_covariate_data = np.array(data_combined_w_covariates[["age"]])
    # array_of_covariate_data.astype(np.float64)
# Adding labels to controls and disease sets  - then combining them
data_disease.insert(0, 'disease/controls', 1)
data_controls.insert(0, 'disease/controls', 0)
data_combined_labelled = pd.concat([data_disease, data_controls])

array_of_clinical_labels = np.array(data_combined_labelled.iloc[:,0])
array_of_data_to_be_adjusted = np.array(data_combined_labelled.iloc[:,1:])
# swap in arrays for your data; shapes of each array as follows:
# array_of_data_to_be_adjusted.shape = (Number of people, Number of biomarkers)
# array_of_clinical_labels.shape = (Number of people)
# array_of_covariate_data.shape = (Number of people, Number of covariates)

# (1d) RUN controls residuals:
data_adjusted = pd.DataFrame(controls_residuals(array_of_data_to_be_adjusted, array_of_clinical_labels, array_of_covariate_data))
print("data_adjusted:", data_adjusted.shape)

# data_adjusted.columns = ["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore"]

# (1e) Re-splitting updated data for step (2)
array_of_clinical_labels = pd.DataFrame(array_of_clinical_labels)
data_with_labels = pd.concat([data_adjusted, array_of_clinical_labels], axis=1) # Combining with labels col so can use labels to split
data_with_labels.columns = ["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "labels"]
data = data_with_labels[data_with_labels['labels'].values == [1]].iloc[:, :5]
data_controls = data_with_labels[data_with_labels['labels'].values == [0]].iloc[:, :5]

# np version
# data_with_labels = np.concatenate((data_adjusted, array_of_clinical_labels), axis=1)
# data = data_adjusted[np.where(data_adjusted[:,0] == 1)]
# data_controls = data_adjusted[np.where(data_adjusted[:,0] == 2)]


# (2) Calculate the mean and standard deviation of each biomarker in your controls dataset
mean_controls = np.mean(data_controls,axis=0)
std_controls = np.std(data_controls,axis=0)


# (3) Z-score your data by taking (data-mean_controls)/std_controls.
# data_zscore = scipy.stats.zscore(data) #  Gets different results...
data = (data-mean_controls)/std_controls
data_controls = (data_controls-mean_controls)/std_controls
print("data:", data.shape)

## (4) Identify any biomarkers that decrease with disease progression, these will have mean_data < mean_controls.
# Multiply the data for these biomarkers by -1.
IS_decreasing = np.mean(data,axis=0)<np.mean(data_controls,axis=0)
print(IS_decreasing)
# data[np.tile(IS_decreasing,(M,1))] = -1*data[np.tile(IS_decreasing,(M,1))]
# data_controls[np.tile(IS_decreasing,(M_controls,1))] = -1*data_controls[np.tile(IS_decreasing,(M_controls,1))]
# NOTES (Peter):
# Depends on score - some get higher, some get lower as diease progresses. Ex brain tissue decreases but fluid goes up.
# Need to look at each Z disease mean compared to controls. If all disease means are positive (above control mean) it's fine.
# If all have negative that's fine. If some are above controls mean and some are below, there's a prob.
# ACTION: If case mean is lower (neg) than case mean, * it by -1
# ACTION: Check mean of all predictors and whether lower/higher than controls means. You need to flip sign so it's going down also.
# motscore(0), mmsetotal(2), exfscore(4)
data = pd.DataFrame((data-mean_controls)/std_controls)
data_controls = pd.DataFrame((data_controls-mean_controls)/std_controls)
data.iloc[:,0] = data.iloc[:,0].mul(-1)
data.iloc[:,2] = data.iloc[:,2].mul(-1)
data.iloc[:,3] = data.iloc[:,3].mul(-1)
data.iloc[:,4] = data.iloc[:,4].mul(-1)
IS_decreasing = np.mean(data,axis=0)<np.mean(data_controls,axis=0)
print(IS_decreasing)


# # ...CHECKS...
#
# # Check that the mean of the controls population is 0 (means covariate regression worked)
# print("\n", 'Mean of disease dataset is','\n', np.mean(data,axis=0))
# # Check that the standard deviation of the whole dataset is greater than 1
# print("\n", 'Standard deviation of disease dataset is','\n', np.std(data,axis=0))
# print("\n", 'Mean of controls is','\n', np.mean(data_controls,axis=0))
# # Check that the standard deviation of the controls population is 1
# print("\n", 'Standard deviation of controls is',' \n', np.std(data_controls,axis=0))
# # Check that the mean of the whole dataset is positive
#
# # print("\n", 'Mean of disease dataset is','\n', round(np.mean(data,axis=0)),2)
# # # Check that the standard deviation of the whole dataset is greater than 1
# # print("\n", 'Standard deviation of disease dataset is','\n', round(np.std(data,axis=0)),2)
# # print("\n", 'Mean of controls is','\n', round(np.mean(data_controls,axis=0)),2)
# # # Check that the standard deviation of the controls population is 1
# # print("\n", 'Standard deviation of controls is',' \n', round(np.std(data_controls,axis=0)),2)
# # # Check that the mean of the whole dataset is positive
#
# # print("\n", "medians of disease =","\n",(data.median(axis=0)))
#
# data_disease.hist()
# data_controls.hist()
#
#
# # ******************* MORE SETTINGS ******************
#
# Z_vals = np.array([[1, 2, 3]] * N)  # Z-scores for each biomarker. This is the set of z-scores you want to include for each biomarker. The more z-scores you use the longer the SuStaIn algorithm will take to run. Z_vals has size N biomarkers by Z z-scores. If you have more z-scores for some biomarkers than others you can simply leave zeros at the end of biomarker rows with fewer z-scores
# Z_max = np.array([5] * N)  # The maximum z-score reached at the end of the progression, with size N biomarkers by 1. I'd suggest choosing a value around the 95th percentile of your data but you can experiment with different values. I typically choose an integer for interpretability but you don't have to
#
# SuStaInLabels = [] # The names of the biomarkers you are using, for plotting purposes.
# for i in range(N):
#     SuStaInLabels.append('Biomarker ' + str(i))  # labels of biomarkers for plotting
#
# N_startpoints = 25 # The number of startpoints to use when fitting the subtypes hierarchichally. I'd suggest using 25.
# N_S_max = 3 # The maximum number of subtypes to fit. I'd suggest starting with a lower number - maybe three - and then increasing that if you're getting a significantly better fit with the maximum number of subtypes. You can judge this roughly from the MCMC plot. To properly evaluate the optimal number of subtypes you need to run cross-validation.
# N_iterations_MCMC = int(1e5) # The number of iterations for the MCMC sampling of the uncertainty in the progression pattern. I'd recommend using 1x10^5 or 1x10^6.
# output_folder = "/Users/lucyrothwell/Google_Drive/MSc_Comp/9_Dissertation HD/SuStaIn-master" # Choose an output folder for the results.
# dataset_name = "Enroll_Sustain_Ouput" # Name the results files outputted by SuStaIn.
#
# sustain_input = ZscoreSustain(data,
#                               Z_vals,
#                               Z_max,
#                               SuStaInLabels,
#                               N_startpoints,
#                               N_S_max,
#                               N_iterations_MCMC,
#                               output_folder,
#                               dataset_name,
#                               False)
#
#
# # ******************* RUN IT ***********************
#
# # runs the sustain algorithm with the inputs set in sustain_input above
# sustain_input.run_sustain_algorithm()