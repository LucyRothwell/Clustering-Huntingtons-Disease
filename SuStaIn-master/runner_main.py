###
# Adapted from:
# pySuStaIn: SuStaIn algorithm in Python (https://www.nature.com/articles/s41468-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
# Contributors: Leon Aksman (l.aksman@ucl.ac.uk), Arman Eshaghi (a.eshaghi@ucl.ac.uk), Alex Young (alexandra.young@kcl.ac.uk)
# Adapted by: Lucy Rothwell
#######################################################################################################################

# CONTENTS
# Items starting ">>>" = for testing new individuals patients
# ---Preprocessing---
# (0) Load data
# (0.1) Recoding the HD category column
# (0.2) OneHotEncoding
# >>> (0.3) Selecting longitudinal_data + new_patient_data for later
# (1) Define settings: Sustain type & features etc
# (2) Fix erroneous values
# >>> (2.5) Holding out new_patient_data to exclude from next step
# (3) Select Baseline / Follow up only
# >>> (3.5) Add new_patient_data back on for processing
# (4) Get rid of medical and demographic features (around 80 columns) + drop columns with >50% NaNs
# (5) Test monotonicity of features (VALIDATION)
# (6) Normality test
# (7) Feature selection
# (8) Features subset
# (9) Deal with missing values
# (10) Remove outliers
# >>> (10.5) Remove and holdout new_patient_data for sustain
# (11) Covariate adjustment(+ Z-scoring if sustainType=ZScoreSustain)
# >>> (11.5) Covariate adjustment for new_patient_data
# --- Analysis ---
# (12) RUN model using preprocessed data
# (13) T-test likelihood differences between subtypes
# (14) T-test CAG distributions between subtypes



# --- Validation ---
# (15) Find optimal number of subtypes (see step 10) - CODE WRITTEN UP
# (16) Find subtype and stage of an individual NEW DATA
#   > Compare this person at Baseline and Visit 2 - do they stay in same subtype / move up a stage



#######################################################################################################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from kde_ebm.mixture_model import fit_all_kde_models, fit_all_gmm_models
from kde_ebm import plotting
from matplotlib import cbook as cbook
import seaborn as sns
# warnings.filterwarnings("ignore", category=cbook.mplDeprecation)
import sklearn.model_selection
from sklearn.preprocessing import OneHotEncoder

from ZscoreSustain import ZscoreSustain
from MixtureSustain import MixtureSustain
from AbstractSustain import AbstractSustain

from pkge0.preprocessing_pipeline import * # Created by author, Lucy Rothwell
from pkge0.feature_selection import * # Created by author, Lucy Rothwell

import time
start_time = time.time()

# (0) Load data

enroll_df = pd.read_csv(os.path.dirname(os.getcwd()) + "/data_enroll_hd/enroll.csv", delimiter=',')
profile_df = pd.read_csv(os.path.dirname(os.getcwd()) + "/data_enroll_hd/profile.csv", delimiter=',')

data = pd.merge(enroll_df, profile_df [ ["sex", "subjid"]], on="subjid", how='left') # Adding in "sex" and "subjid" from profile.csv

# (0.1) Recoding the HD category column
data = re_code_hdcat(data)  # Function written by author in "preprocessing_pipeline.py"

# (0.2) OneHotEncoding
# One hot encodoing of "sex" (and "siteID"?)
data = data [data ['sex'].notna()]
sex = np.asarray(data ["sex"])
sex = sex.reshape(-1,1)
# define one hot encoding
encoder = OneHotEncoder(sparse=False)
# transform data
onehot = encoder.fit_transform(sex)
# Replacing the "sex" column with "female" and "male"
data.drop("sex", inplace=True, axis=1)
data.insert(1, "female", onehot [:,0])
data.insert(1, "male", onehot [:,1])

# # ------------------ Histogram to check spread ----------------------------
# testing_var = "height"
# df = data[data[testing_var] > -50]
# # df = df [df ["hdcat"]==0]
# df = df [df [testing_var] < 150]
# tms = df [testing_var].values
# hist =  []
# labels =  []
# for i in np.unique(df ['hdcat'].values):
#     if np.isnan(tms [df ['hdcat'].values==i]).all():
#         continue
#     hist.append(tms [df ['hdcat'].values==i])
#     labels.append(str(i))
# plt.hist(hist,label=labels,stacked=True)
# plt.yscale("log")
# plt.xlabel(testing_var)
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()
# # ------------------------------------------------------------------------

## (1) Settings: Define Sustain type & features etc

# --SUBMITTED DISSERTATION:--
# features_list =  ["motscore", "chorrue", "sacinith", "sdmt2", "verfct7", "trlb3", "depscore", "irascore", "pbas3sv"] # Features for analysis (clinical test scores)

# -- DATA DRIVEN -- (exlucding motscore
# features_list = ["chorlue", "sacinith", "fingtapl", "prosupl", "scnt1", "swrt1", "irascore", "pbas3sv"] # Features for analysis (clinical test scores)
# features_list = ["chorlue", "fingtapl", "scnt1", "swrt1", "pbas3sv", "irascore"] # Features for analysis (clinical test scores)
features_list = ["chorlue", "fingtapl", "scnt1", "sdmt1", "pbas3sv", "pbas6sv"] # Features for analysis (clinical test scores)

# --COMPOSITE SCORES:--
## features_list =  ["motscore", "mmsetotal", "sdmt2", "verfct7", "trlb3", "depscore", "irascore", "anxscore"] # COMPOSITE (1) ---- MOTOR=1 COMPISITE, COG= 1 COMPOSITE + 3 TOTAL ERRORS. PSYCH=ALL COMPOSITE
## features_list =  ["motscore", "mmsetotal", "depscore", "irascore", "anxscore"] # COMPOSITE (2) ---- MOTOR=1 composite. COG=1 COMPOSITE. PSYCH=3 COMPOSITE
# features_list =  ["motscore", "brady", "scnt1", "sdmt1", "pbas6sv", "irascore", "tfcscore"] # Composites

# # -- CLINICALLY ADVISED ---
# features_list = ["motscore", "sacinith", "brady", "scnt1", "sdmt1", "pbas6sv", "irascore"] # Clinically advised features

# Clinically advised
# - total motor score (M)
# - dystonia (M)
# - total chorea (M)
# - bradykinetic (M)
# - saccade (M)
# - symbol digit modality (C)
# - stroop test (C)

# More likely to be monotonic
# - apathy (P)
# - irritability (P)

# Top Random Forest
# - diagconf
# - indepscl - Subject's independence in %
# - miscore (M)
# - chorrue (M) **
# - motscore (M)
# - fingtapl (M) **
# - sacinith (M) **
# - chorlue (M) **
# - fiscore: UHDRS Functional score incomplete
# - scnt1 (C): ** stroop correct
# - sacinitv: Group saccade initiatitation vertical
# - fingtapr (M) **
# - tfcscore **
# - fascore: UHDRS Functional assessment score
# - prosupl (M): ** Group Pronate supinate‐hands left
# - swrt1 (C)

covariates_list =  ["age", "male", "female", "isced"] # The effects of these will be regressed out later
subj_hdcat =  ["subjid", "hdcat"] # These will be used throughout - 'subjid' as primary key and 'hdcat' as labels
remove =  ["studyid", "seq", "visit", "visdy", "visstat"] # Non-numerical features to remove
visit_type = "Baseline" # We only want baseline visits
sustainType = "mixture_KDE" # for use in functions later

N_S_max = 5 # NUMBER OF SUBTYPES
N_folds = 10 # NUMBER OF FOLDS FOR CROSS VALIDATION

# ------------------------------------ PLOT - EXPLORE VARS ------------------------------------------------------------
# data ["motscore"].hist()
# sns.distplot(data ['motscore'], hist=True, kde=True,
#              bins=int(180/5), color = 'darkblue',
#              hist_kws={'edgecolor':'black'},
#              kde_kws={'linewidth': 4})

# # ***NEEDS FIXED*** Replicating isced value on follow up visits - TO GET BIGGER DATA SET (isced has most missing as is only recorded on Baseline)
# data["isced"].isnull().sum() # Check num of missing isced values
#
# def replicate_isced_vals(data):
#     start_time2 = time.time()
#     for index, row in data.iterrows(): # *** PROBLEM!
#         if row["visit"] == "Baseline":
#             isced_val = row["isced"]
#             subj_id = row["subjid"]
#             for index, row in data.iterrows():
#                 if row["subjid"]==subj_id and row["visit"] != "Baseline":
#                     row["isced"] = isced_val
#                     print("row[isced]=", row["isced"])
#     print("--- %s seconds ---" % (time.time() - start_time2))
#
# data["isced"].isnull().sum() # Check new num of missing isced values

# --------------------------------------------------------------------------------

num_covars = len(covariates_list)
num_predictors = len(features_list)
skip_start = len(subj_hdcat)

# # >>> (1.1) Finding best longitudinal subjects to use (by number of visits)
# max_rows = 0
# visits = 0
# for i in range(4,8): # Returns df of visits with highest # rows but lowest NaN counts (for feature-selected subsets). (Range refers to number of visits)
#     counts = data['subjid'].value_counts()
#     long_data = data[data['subjid'].isin(counts.index[counts == i])] # Taking subset of patients with i visits
#     long_data_feat = long_data[subj_hdcat + features_list + covariates_list + ["seq"]] # Taking features of that subset
#     print(i, "before NaN drop:", long_data_feat.shape)
#     print("\n","NaN values,",i,"visits:","\n",long_data_feat.isnull().sum())
#     # long_data_feat_drop = replicate_isced_vals(long_data_feat)
#     long_data_feat_drop = long_data_feat.dropna() # Dropping NaNs from that subset
#     new_counts = long_data_feat_drop['subjid'].value_counts()
#     long_data_new = long_data_feat_drop[long_data_feat_drop['subjid'].isin(new_counts.index[new_counts == i])] # Re-taking subset of patients with i viist now that NaNs have been dropped
#     print(i, "after NaN drop:", long_data_new.shape, "\n")
#     if long_data_new.shape[0] > max_rows: # If long_data_new has more rows than a previous subset...
#         max_rows = long_data_new.shape[0]
#         longitudinal_data = long_data_new # This is the output at end of loop
#         visits = i
# print("Number of visits with most subjects after NaN drop =", visits)
#
#
# # >>> (1.2) Selecting indivuduals from longitudinal data
# longitudinal_de_duped = longitudinal_data["subjid"].drop_duplicates() # De-duping subjectID for random sampling
# new_patient_id0, new_patient_id1, new_patient_id2  = longitudinal_de_duped.sample(n=3, replace=False) # Sampling patients with which to test
#
# new_patient_data0 = longitudinal_data[longitudinal_data["subjid"] == new_patient_id0]
# new_patient_data1 = longitudinal_data[longitudinal_data["subjid"] == new_patient_id1]
# new_patient_data2 = longitudinal_data[longitudinal_data["subjid"] == new_patient_id2]
#
# # Check sequence of visits is correct
# new0_to_check_nocovaradj = longitudinal_data[longitudinal_data["subjid"] == new_patient_id0]
# new1_to_check_nocovaradj = longitudinal_data[longitudinal_data["subjid"] == new_patient_id1]
# new2_to_check_nocovaradj = longitudinal_data[longitudinal_data["subjid"] == new_patient_id2]
#
# new_patients = new_patient_id0, new_patient_id1, new_patient_id2 # For use later


# (2) Fix erroneous values
data = fix_erroneous(data)

## -----------------------------------------------------------------
## >>> (2.1) Holding out all of the individuals (new_patient's) data to exclude from training

# data = data[(data["subjid"] != new_patients[0]) & (data["subjid"] != new_patients[1]) & (data["subjid"] != new_patients[2])] # Removing that data from training set
#
# # ------------------------------------------------------------------
#
# # (3) Select Baseline / Follow up only
# data = data [data ['visit'].values ==  [visit_type]] # Selecting BASELINE visits only
# data.drop_duplicates(subset="subjid", inplace=True, keep="first") # Removing duplicates
#
# # >>> (3.5) Add on individuals longitudinal data for processing
# data = pd.concat([data, new_patient_data0, new_patient_data1, new_patient_data2], axis=0)

# (4) Get rid of unwanted cols - make rest numeric
data.drop(remove, axis=1, inplace=True) # > Unwanted columns set at beginning
# data.columns.get_loc("motscore")
isced = data["isced"] # Isolating covariate isced before dropping rest
data = data.drop(columns=data.columns[5:84]) # # Medical history / demographic features etc (leaving only test-related features)
data.insert(5, "isced", isced) # adding isced back in
# > Make all values numeric
for column in data.iloc [:, skip_start:]: # Except subject id, hd_cat
    data [column] = pd.to_numeric(data [column])
# > Remove cols with > 40% NaNs
# print("Columns with > 40% NaNs (which have been dropped):")
# for column in data: # -1 to exclude isced *** This exclused isced unless done just on Baseline visits)
#     if (data [column].isna().sum() / data.shape [0]) > 0.4:
#         # print(column)
#         data.drop(column, inplace=True, axis=1)

# # # (5) Test for monotinicity of variables
# # Linear regression
# from sklearn.linear_model import LinearRegression
# X = np.array(longitudinal_data["motscore"])
# X = X.reshape(-1, 1)
# y = np.array(data["diagconf"])
# y = y.reshape(-1, 1)
# reg = LinearRegression().fit(X, y)
# reg.score(X, y)
# print("reg.coef_:", reg.coef_)
# print("reg.intercept_:", reg.intercept_)
# print("reg.predict:", reg.predict(np.array([[3, 5]])))
#

# # (6) Normality test
# data_noNan = missing_vals(data, print_miss_counts=True, method="mean")
# for column in data_noNan.iloc [:, skip_start:]: # skipping subjid, hdcat
#     title = column
#     column = data_noNan [column]
#     stat, p = scipy.stats.normaltest(column)
#     print(title, round(stat, 4), round(p,8))

# (5) Test monotonicity of features
# > Linear regression comparing <feature> with diagconf or tfc scores
# > Use “CAP score” with longitudinal data (formula for it)
# > Do simple regression of that marker against cap score for each person

# cap_column = []
#
# data_monot_test = pd.concat((data[features_list], data["diagconf"]), axis=1).dropna()
#
# diag_conf = np.asarray(data_monot_test["diagconf"]).reshape(np.asarray(data_monot_test["diagconf"]).shape[0],-1)
#
# for i in features_list:
#     j = np.asarray(data_monot_test[i]).reshape(np.asarray(data_monot_test[i]).shape[0], -1)
#     lr = linear_model.LinearRegression()
#     lr.fit(j, diag_conf)
    # print(i, "coefficient:", lr.coef_, "intercept:", lr.intercept_)

# -----------------------------------------------------
# # Alternative monotinicity test?
# def non_increasing(L):
#     return all(x>=y for x, y in zip(L, L[1:]))
#
# def non_decreasing(L):
#     return all(x<=y for x, y in zip(L, L[1:]))
#
# def monotonic(L):
#     return non_increasing(L) or non_decreasing(L)
#
# monotonic(data_monot_test)
#
# for i in features_list:
#     monotonic(data_monot_test[i])
# -----------------------------------------------------

# # (7) Feature selection
# t_test_results = feature_selection(data.iloc [:, 1:], "hdcat", method="t_test", sign_lev=0.05) # 1 to skip subjID
# wilcoxon_results = feature_selection(data.iloc [:, 1:], "hdcat", method="wilc", sign_lev=0.05) # 1 to skip subjID
# mann_whitney_results = feature_selection(data.iloc [:, 1:], "hdcat", method="mann_w", sign_lev=0.05)
# rf_importances = feature_selection_rf(data.iloc [:, 1:], "hdcat")

# (8) Features subset w covars (plus subjID and hdcat)
data_subj_hdcat = data [subj_hdcat]
data_features = data [features_list]
data_covariates = data [covariates_list]
data = pd.concat( [data_subj_hdcat, data_features, data_covariates], axis=1)

# data [data ["hdcat"]==1].hist()

# (9) Deal with missing values
data = missing_vals(data, method="drop")


# (10) Remove outliers
# > Remove outliers 5*SDs from mean   *** PROBLEM - removes almost all mmse rows
disease = data [data ["hdcat"]==1]
controls = data [data ["hdcat"]==0]
preman = data [data ["hdcat"]==2]
if sustainType == 'mixture_GMM' or sustainType == "mixture_KDE": # Include hdcat (a few extra steps for this)
    print("\n""features_list_cov =", features_list)
    print("\n""DISEASE", "\n")
    disease = outlier_removal(disease, features_list, sd_num=5)
    print("\n""CONTROLS", "\n",)
    controls = outlier_removal(controls, features_list, sd_num=5)
    print("\n""PREMAN")
    preman = outlier_removal(preman, features_list, sd_num=5)
    data = pd.concat( [disease, controls, preman])
elif sustainType == 'zscore': # Exclude hdcat.
    data = outlier_removal(data, features_list, sd_num=5) # 1 for subjID still in

# features_list = features_list [:-1] # removing hdcat from features list (comes out of outlier_removal() with hdcat)

# dis_cont = data [data ["hdcat"] != 2]

# # # ---------(Counting analysis data for write-up)--------------
# Merge data to get data counts
data_m = pd.DataFrame(data)
data_merge = pd.merge(data_m, profile_df, on="subjid", how='left')
data_merge_disease = data_merge[data_merge["hdcat"] == 1]
data_merge_preman = data_merge[data_merge["hdcat"] == 2]

data_disease_pm = pd.concat([data_merge_disease, data_merge_preman], axis=0)
data_merge_controls = data_merge[data_merge["hdcat"] == 0]

print("--- COHORT STATS - DISEASE ---")
print(data_disease_pm["hdcat"].value_counts())
print(data_disease_pm["hdcat"].value_counts(normalize=True)*100)
print(data_disease_pm["sex"].value_counts())
print(data_disease_pm["sex"].value_counts(normalize=True)*100)
print(data_disease_pm["race"].value_counts())
print(data_disease_pm["race"].value_counts(normalize=True)*100)
print(data_disease_pm["isced"].value_counts())
print(data_disease_pm["isced"].value_counts(normalize=True)*100)
print(data_disease_pm["age"].mean())

print("--- COHORT STATS - CONTROL ---")
print(data_merge_controls["hdcat"].value_counts())
print(data_merge_controls["hdcat"].value_counts(normalize=True)*100)
print(data_merge_controls["sex"].value_counts())
print(data_merge_controls["sex"].value_counts(normalize=True)*100)
print(data_merge_controls["race"].value_counts())
print(data_merge_controls["race"].value_counts(normalize=True)*100)
print(data_merge_controls["isced"].value_counts())
print(data_merge_controls["isced"].value_counts(normalize=True)*100)
print("Mean age = ", data_merge_controls["age"].mean())

# Getting CAG array to concatenate to output later
data_merge_no_preman = data_merge [data_merge ["hdcat"] != 2]
data_merge_no_preman["caghigh"].replace(">70", 71, inplace=True)
data_merge_no_preman["caglow"].replace(">28", 29, inplace=True)
caghigh = pd.to_numeric(data_merge_no_preman ["caghigh"])
caglow = pd.to_numeric(data_merge_no_preman ["caglow"])
# type(caghigh.iloc [0])
row = 0
for i in caghigh:
    if i == 1:
        caghigh.iloc [row] = caglow.iloc [row]
    row = row + 1
cag = caghigh
# # ---------------------------------------------------------------

# >>> (10.5) Isolate 'subjid' and 'visit' for replacement after covariate adjustment
# Isolate subjid to put back on later
subjid_col = data['subjid']
subjid_col = subjid_col.reset_index()
subjid_col = subjid_col.iloc[:, 1]
subjid_col = pd.DataFrame(subjid_col.values.reshape(-1, 1)) # Turns it to a np array

# Joining visit col back on so can select by visit type
subjid_visit = enroll_df[["subjid", "visit"]]
data_n = data.join(subjid_visit, lsuffix="", rsuffix="_drop", how='left')
data = data_n.drop("subjid_drop", axis=1)

# Isolate visit to put back on later
visit_col = data["visit"] # Selecting BASELINE visits only
# visit_col.drop_duplicates(subset="subjid", inplace=True, keep="first") # Removing duplicates

# Dropping these cols for covariate adjustment
data = data.drop(["subjid","visit"], axis=1)

# ----------------------------------------------------------------------------------------------------------------------
# (11) Covariate adjustment(+ Z-scoring if sustainType=ZScoreSustain)
# Regresses out covariates of predictors + if ZScoreSustain; (2) Finds mean and SD of features, (3) Takes Z scores, (4) Makes sure all Z scores are increasing
data = covariate_adjustment(data, covariates_list, features_list, sustainType)
# Check adjustment has worked - i.e., mean of each control biomarker == 0.
feat = 0
for column in data [data [:, len(features_list)] == 0].T:  # PROBLEM   8 = hdcat. Check mean of each control biomarker now == 0 (means adjustment has worked)
    mean = np.mean(column)
    print("\n control mean, biomarker", feat, "=", np.around(mean, 4))
    feat = feat + 1
# #
# # **** Do this if not doing covariate adjustemnt - to get rid of covariates ****
# data = pd.DataFrame(data) # Turn to dataframe to allow the following operations
# hdcat = data["hdcat"]
# predictors = data[features_list]
# # predictors = data.iloc [:, 1:len(features_list)-num_covars-1] # 1 to exclude subjid, 2 covariates and hdcat
# data = pd.concat([predictors, hdcat], axis=1)

# ----------------------------------------------------------------------------------------------------------------------
# ------------- HISTS CHECK ------------------------
# data_hists = pd.DataFrame(data)
# data_hists = data_hists.iloc [:, :-1]
# data_hists.columns =  ["motscore", "chorrue", "sacinith", "sdmt2", "verfct7", "trlb3", "depscore", "irascore", "pbas3sv"]
#
# # data_hist [data_hist.iloc [:, -1] == 1].hist()
# # data_hist [data_hist.iloc [:, -1] == 0].hist()
# # print(motscore.value_counts())
# data_hists.hist()
# -------------------------------------------------

# ---------------- for testing single inidividuals only --------------------
# # >>> (11.5) Remove and holdout longitudinal data for sustain
# data = pd.DataFrame(data) # Change from array to df
# data = pd.concat([data, subjid_col], axis=1, ignore_index=True) # Put subjID col back on so can select new_patient_id
# feat_list_plus = features_list.copy()
# feat_list_plus.append("hdcat") # Adding hdact to column labels
# feat_list_plus.append("subjid") # Adding subjid to column labels
# data.columns = feat_list_plus # Putting column labels back on
#
# new_patient_data0 = data[data["subjid"] == new_patient_id0] # Selecting new patient data
# new_patient_data1 = data[data["subjid"] == new_patient_id1] # Selecting new patient data
# new_patient_data2 = data[data["subjid"] == new_patient_id2] # Selecting new patient data
#
# columns = data.columns
# # subjid_col = data["subjid"]

## Deleting new patient data from training set
# data = data[data["subjid"] != (data["subjid"] != new_patients[0]) & (data["subjid"] != new_patients[1]) & (data["subjid"] != new_patients[2])]
# data = data.iloc[:, :-1] # Removing subj_id column again
# data = np.array(data) # Converting back to array for main()
# ---------------- for testing single inidividuals only --------------------

# Add back 'visit' column + column headers
data = pd.DataFrame(data)
visit_col = pd.DataFrame(visit_col.values.reshape(-1, 1))
data = pd.concat([data, visit_col], axis=1)
data.columns = [features_list + ["hdcat"] + ["visit"]]

# Define longitudinal data
data_all_long = data.drop("visit", axis=1)

# Select baseline only (and then remove visit column)
data_baseline = data[data.iloc[:, -1] == visit_type].drop("visit", axis=1) ## -1 = visit column (doesn't work using column title for some reason)
data_baseline = np.array(data_baseline) # Changing to array for main()


# ------------------------end of pre-processing-------------------------------------------------------------------------

# (12) RUN model using preprocessed data
def main(data, sustainType):

# ----SETTINGS----
    # cross-validation
    validate = True
    # N_folds = 10 # SET AT TOP INSTEAD. recommended 10

    if sustainType == "mixture_KDE" or "mixture_GMM":
        N = data.shape [1]-1  # number of biomarkers. If using kdemm - hdcat is still in data so we need to -1 to get num of biomarkers
    if sustainType == "zscore":
        N = data.shape [1]  # number of biomarkers

    M = data.shape [0]  # number of observations ( e.g. subjects )

    N_startpoints = 25 # Number of starting points to use when fitting the subtypes hierarchichally. I'd suggest using 25.
    # N_S_max = 3 # SET AT TOP INSTEAD. The maximum number of subtypes to fit. I'd suggest starting with a lower number - maybe three - and then increasing that if you're getting a significantly better fit with the maximum number of subtypes. You can judge this roughly from the MCMC plot. To properly evaluate the optimal number of subtypes you need to run cross-validation.

    # N_iterations_MCMC = int(1e5)  # The number of iterations for the MCMC sampling of the uncertainty in the progression pattern. I'd recommend using 1x10^5 or 1x10^6.
    N_iterations_MCMC = 100  # Use this one to make programme quicker on test runs. CHANGE back toint(1e5) for final run.

    # either 'mixture_GMM' or 'mixture_KDE' or 'zscore'
    sustainType = sustainType

    assert sustainType in ("mixture_GMM", "mixture_KDE", "zscore"), "sustainType should be either mixture_GMM, mixture_KDE or zscore"

    dataset_name = 'results'
    output_folder = dataset_name + '_' + sustainType

    Z_vals = np.array( [ [1, 2, 3]] * N)  # Z-scores for each biomarker
    Z_max = np.array( [5] * N)  # maximum z-score

    SuStaInLabels = ["Motor: Finger tap", "Motor: Chorea", "Cogn: Stroop", "Cogn: SDMT", "Psych: Apathy",
                 "Psych: Anxiety"]  # *** Lucy hardcoded for diagram

    # SuStaInLabels =  [] # Used to label plots with feature names later
    # SuStaInStageLabels =  [] # What is this used for?

    # for i in features_list: # Adding the biomarker names
    #     SuStaInLabels.append(i)

# ---- SMOOTHING THE DATA / GETTING KDE FITS -----
# ---- + loading sustain = MixtureSustain()----

    if sustainType == 'mixture_GMM' or sustainType == "mixture_KDE":

        case_control_all = data [data [:, -1] != 2] # Removing pre-manifest - leaving cases and controls (KDE is about drawing out differences between control and disease - so including pre-man would just confuse it)
        labels_case_control = case_control_all [:, -1] # getting labels (-1 to remove hdcat)
        data_case_control = case_control_all [:, :-1] # getting predictors (-1 to remove hdcat)
        data = data_case_control # just renaming for rest of algo

        N = data.shape [1]

        print("SuStaInLabels", SuStaInLabels)

# ---- FITTING THE FEATURE DATA ---- (finding a pdf/LF for each feature - either with GMM or KDE)
    # Estimating the parameters (mean and variance) of the Gaussian (GMM) or estimate the non-parametric (KDE)
    # Finding a 'cluster' for disease and a 'cluster' for controls - Pete
    # 'mixtures' = the set of pdfs for the features
        if sustainType == "mixture_GMM":
            mixtures = fit_all_gmm_models(data, labels_case_control.astype(int))
        elif sustainType == "mixture_KDE":
            mixtures = fit_all_kde_models(data, labels_case_control.astype(int))

        outDir = os.getcwd() # defining where results will go

        # Produces the KDE fits figure
        fig, ax = plotting.mixture_model_grid(data_case_control, labels_case_control, mixtures, SuStaInLabels)
        fig.show()
        fig.savefig(os.path.join(outDir, 'kde_fits.png')) #outDir = output_folder

        # Creating null dataframes (same shape as data) in which to load...
        L_yes = np.zeros(data.shape) # ...the probability that each value in dataframe is DISEASE (across all biomarkers)
        L_no = np.zeros(data.shape) # ...the probability that each value in dataframe is HEALTHY (across all biomarkers)

        # Calculating data likelihood function for each biomarker, N.
        # For each column / feature
        # (1) Use pdf() function on the fits ('mixtures') >> (?) What does this do?
        # (2) In L_yes add all values for that feature that have likelihood of being abnormal. Same for L_no but with health values.
        # The L_yes and L_no dataframes are then used in MixtureSuStain

        # For each biomarker, N, find a pdf for 'healthy' and a pdf for 'disease'. So that for any value of a biomarker,
        # we can determine whether it's disease or no disease

        # (?) Difference 'mixture' fits and mixture pdfs?
        # Filling dataframes L_yes and L_no (each the shape 'data') with a probability value for each cell
        for i in range(N): # For each biomarker
            if sustainType == "mixture_GMM":
                L_no [:, i], L_yes [:, i] = mixtures[i].pdf(None, data [:, i]) # pdf() = function in kde_ebm. Returns controls_score, patholog_score (probability values)
            elif sustainType == "mixture_KDE":
                L_no [:, i], L_yes [:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1)) # CALCULATING PDF

        # Creating MixtureSustain() object

        sustain = MixtureSustain(L_yes, L_no, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder,
                                 dataset_name, use_parallel_startpoints=True)
        # # Plotting pdf...Lucy
        # plt.plot(L_yes [:,1])
        # plt.show()
        #
        # # Trying again...Lucy
        # kde = scipy.stats.gaussian_kde(np.array(L_yes [:,1]))
        # #visualize KDE
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # x_eval = np.linspace(-.2, .2, num=200)
        # ax.plot(x_eval, kde(x_eval), 'k-')
        # #get probability
        # kde.integrate_box_1d( 0.3, np.inf)

    elif sustainType == 'zscore':

        sustain = ZscoreSustain(data, Z_vals, Z_max, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC,
                                output_folder, dataset_name, False)

# ---- RUNNING SUSTAIN ON MixtureSustain() or ZScoreSustain() objects ----
    ml_subtype, ml_stage, samples_sequence = sustain.run_sustain_algorithm()

# ---- CROSS VALIDATION ----
    if validate: # DO CROSS VALIDATION. A way of optimising for number of clusters.
        # Output here (cross_validated_sequences) will be messier. Check there is differences between 1 cluster, 2 clusters etc.
        test_idxs = []

        cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
        # cv_it = cv.split(data, labels)
        cv_it = cv.split(data, labels_case_control.astype(int))

        for train, test in cv_it:
            test_idxs.append(test)
        test_idxs = np.array(test_idxs)

        loglike_matrix, CVIC, log_likelihoods, samples_f = sustain.cross_validate_sustain_model(test_idxs)

        # this part estimates cross-validated positional variance diagrams
        for i in range(N_S_max):
            sustain.combine_cross_validated_sequences(i + 1, N_folds)

    plt.show()
    return L_yes, L_no, log_likelihoods, ml_subtype, ml_stage, sustain, samples_f, samples_sequence, mixtures


if __name__ == '__main__':
    np.random.seed(42)
    # main(data, sustainType)  # (Original. Replaced with line below).
    L_yes, L_no, log_likelihoods, ml_subtype, ml_stage, sustain, samples_f, samples_sequence, mixtures = main(data_baseline, sustainType)


## (13) T-test differences between subtypes

p_values_t_tests = []
for i in range(0, N_S_max-1):
    print("\n subtypes",i,"&",i+1,":", ttest_ind(log_likelihoods[i], log_likelihoods[i+1]))
    stat, p_val = ttest_ind(log_likelihoods[i], log_likelihoods[i+1])
    p_values_t_tests.append(float(p_val))

# t_test_12, p_val_12 = ttest_ind(log_likelihoods [0], log_likelihoods [1])
# t_test_23, p_val_23 = ttest_ind(log_likelihoods [1], log_likelihoods [2])
# t_test_34, p_val_34 = ttest_ind(log_likelihoods [2], log_likelihoods [3])
# t_test_45, p_val_45 = ttest_ind(log_likelihoods [3], log_likelihoods [4])
#
# p_val_12 = round(int(p_val_12), 4)
# p_val_23 = round(int(p_val_23), 4)


#### (14) T-tests between each subtype’s CAG distribution

ml_subtype = pd.DataFrame(ml_subtype)
ml_subtype_wCAG = pd.concat( [ml_subtype, cag], axis=1)
# ml_subtype_wCAG = ml_subtype_wCAG.dropna()

# print("\n" ml_subtype_wCAG.iloc [:, 0].value_counts(normalize=True)*100)

sub1_cag = ml_subtype_wCAG.iloc [:, 1] [ml_subtype_wCAG.iloc [:, 0]==0]
sub2_cag = ml_subtype_wCAG.iloc [:, 1] [ml_subtype_wCAG.iloc [:, 0]==1]
sub3_cag = ml_subtype_wCAG.iloc [:, 1] [ml_subtype_wCAG.iloc [:, 0]==2]

print("Cag S1&2", ttest_ind(sub1_cag, sub2_cag, nan_policy="omit"))
print("Cag S2&3", ttest_ind(sub2_cag, sub3_cag, nan_policy="omit"))
print("Cag S1&3", ttest_ind(sub1_cag, sub3_cag, nan_policy="omit"))

print("--- %s mins ---" % ((time.time() - start_time)/60))


#### (15) Subtypes and stages of all participants in training set

# Forming array with subtypes and stages of all participants in training set
results_subtype_stage_all = np.concatenate([ml_subtype, ml_stage], axis=1)
results_subtype_stage_all = pd.DataFrame(results_subtype_stage_all)
results_subtype_stage_all = results_subtype_stage_all.astype(int)

# Distributions of stages for each cluster
stage_distributions = {}

# for i in range(0, N_S_max+1):
#     distribution = pd.DataFrame(results_subtype_stage_all[results_subtype_stage_all.iloc[:, 0] == i]) # 0 = subtype column
#     stage_distributions["Subtype", i] = distribution.iloc[:, 1] # 1 = stage column
#     plt.hist(stage_distributions["Subtype", i])

# for i in range(1,N_S_max): # For each subtype
#     figs, axs = plt.subplots()
#     col = pd.DataFrame(results_subtype_stage_all[results_subtype_stage_all.iloc[:, 0] == i]) # Create a subset ("col") where subtype = i (so we have a (n, 2) array where columns = subtype and stage (0 = subtype column)
#     stage_distributions["Subtype", i] = col.iloc[:, 1] # Create a (n, 1) array showing only stages (for subtype i)
#     axs.hist(stage_distributions["Subtype", i], bins=len(features_list), range=(1,6), x_label="Stages", y_label="Number of participants in sample")
#     plot_title = 'Stage dist for Subtype', i
#     axs.set_title(plot_title)

# TESTING CODE - use test_plotting.py code to fix this
for i in range(0,N_S_max): # For each subtype
    figs, axs = plt.subplots()
    col = pd.DataFrame(results_subtype_stage_all[results_subtype_stage_all.iloc[:, 0] == i]) # Create a subset ("col") where subtype = i (so we have a (n, 2) array where columns = subtype and stage (0 = subtype column)
    stage_distributions["Subtype", i] = col.iloc[:, 1] # Create a (n, 1) array showing only stages (for subtype i)
    axs.hist(stage_distributions["Subtype", i], bins=len(features_list), range=(1,6))
    # ax1 = axs.add_subplot(211)
    axs.set_ylabel('Number of participants in sample')
    # ax2 = i.add_subplot(211)
    axs.set_xlabel('Stages')
    plot_title = 'Stage dist for Subtype', i+1
    axs.set_title(plot_title)
    # *** Add plots to big plot

# data_baseline = data[data.iloc[:, -1] == visit_type]

### (16) Find optimal number of subtypes
# > See step 10 - just increase N_sub at beginning

for i in range(1,N_S_max+1):
    plt.scatter([i,i,i,i,i,i,i,i,i,i], log_likelihoods[i-1]) # y_label="Log likelihood", x_label="Number of subtypes in model"

### (17) Find subtype and stage of an individual NEW DATA
# > At Visit 1 (Baseline) and Visit 2. Do they stay in same subtype / move up a stage?

# get_subtype_stage(new_patient_data0, features_list, sustainType, mixtures, sustain, samples_sequence, samples_f, N_samples=1000)
#
# get_subtype_stage(new_patient_data1, features_list, sustainType, mixtures, sustain, samples_sequence, samples_f, N_samples=1000)
#
# get_subtype_stage(new_patient_data2, features_list, sustainType, mixtures, sustain, samples_sequence, samples_f, N_samples=1000)


# ### (18) Find subtype and stage - all patients
#
# # *** NOTE *** (do onlu requested by journal)
# # Peter's suggested approach:
# # Do two runs of functions:
# # o	(1) subtype_inds() - get array of baseline
# # o	(2) subtype_inds_new_data() - get array of follow up
# # So end up with two identially sized arrays - one for baseline and one for visit X. Then we can see if subtypoe and stage moves as expected between first and second arrays
#
# # Getting df ready (putting subjid col back on plus adding headers and changing to df)
# data = np.concatenate((data_all_long, subjid_col), axis=1)
# data = pd.DataFrame(data)
# data.columns = [features_list + ["hdcat", "subjid"]]
# data = data.drop("hdcat", axis=1) # Removing hdcat
#
# columns_subtyped_all = ["subjid", "V1 subtype", "V1 subtype prob", "V1 stage", "V1 stage prob", "V2 subtype", "V2 subtype prob", "V2 stage", "V2 stage prob"]
# data_subtyped_all = pd.DataFrame(columns=columns_subtyped_all) # COLUMNS NOT INSERTED HERE. Creating empty dataframe for results
# # data_subtyped_all.columns = columns_subtyped_all
#
# from pkge0.preprocessing_pipeline import * # Created by author, Lucy Rothwell
# from pkge0.feature_selection import * # Created by author, Lucy Rothwell
#
# # Running get_subtype_stage() for every row in data
# # Vectorize this loop to improve speed? https://stackoverflow.com/questions/1422149/what-is-vectorization#:~:text=%22Vectorization%22%20(simplified)%20is,array%20simultaneously%20N%2F4%20times.
# for row_num in range (0, len(data)+1): # Iterating over each row
#     print("row_num:", row_num)
#     row_num = 1 #Testing
#     new_patient_data = pd.DataFrame(data.iloc[row_num, :].values.reshape(1,-1))
#     new_patient_data.columns = features_list + ["subjid"]
#     print("new_patient_data.shape", new_patient_data.shape)
#     ml_subtype_v1, prob_ml_subtype_v1, ml_stage_v1, prob_ml_stage_v1, ml_subtype_v2, prob_ml_subtype_v2, ml_stage_v2, prob_ml_stage_v2 = get_subtype_stage(new_patient_data, features_list, sustainType, mixtures, sustain, samples_sequence, samples_f, N_samples=1000)
#     data_subtyped_all.append([ml_subtype_v1, prob_ml_subtype_v1, ml_stage_v1, prob_ml_stage_v1, ml_subtype_v2, prob_ml_subtype_v2, ml_stage_v2, prob_ml_stage_v2])

#### (19) Compare this progression estimation with longitudinal data to see if predicted stage is same as actual stage
# > How to measure actual stage?
