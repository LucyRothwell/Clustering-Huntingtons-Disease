###
# Adapted from:
# pySuStaIn: SuStaIn algorithm in Python (https://www.nature.com/articles/s41468-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
# Contributors: Leon Aksman (l.aksman@ucl.ac.uk), Arman Eshaghi (a.eshaghi@ucl.ac.uk), Alex Young (alexandra.young@kcl.ac.uk)
# Adapted by: Lucy Rothwell
#######################################################################################################################

# CONTENTS
# ---Preprocessing---
# (0) Load data
# (0.1) Recoding the HD category column
# (0.2) OneHotEncoding
# (1) Define settings: Sustain type & features etc
# (2) Fix erroneous values
# (3) Select Baseline / Follow up only
# (4) Get rid of medical and demographic features (around 80 columns) + drop columns with >50% NaNs
# (5) Test monotonicity of features
# (6) Normality test
# (7) Feature analysis
# (8) Features subset
# (9) Deal with missing values
# (10) Remove outliers
# (11) Data counts for write-up
# (12) Covariate adjustment(+ Z-scoring if sustainType=ZScoreSustain)
# --- Analysis ---
# (13) RUN model using preprocessed data
# (14) T-test likelihood differences between subtypes
# (15) T-test CAG distributions between subtypes


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

## (1) Settings: Define Sustain type & features etc
features_list = ["chorlue", "fingtapl", "scnt1", "sdmt1", "pbas3sv", "pbas6sv"] # Features for analysis (clinical test scores)
covariates_list =  ["age", "male", "female", "isced"] # The effects of these will be regressed out later
subj_hdcat =  ["subjid", "hdcat"] # These will be used throughout - 'subjid' as primary key and 'hdcat' as labels
remove =  ["studyid", "seq", "visit", "visdy", "visstat"] # Non-numerical features to remove
visit_type = "Baseline" # We only want baseline visits
sustainType = "mixture_KDE" # for use in functions later

N_S_max = 5 # NUMBER OF SUBTYPES
N_folds = 10 # NUMBER OF FOLDS FOR CROSS VALIDATION
num_covars = len(covariates_list)
num_predictors = len(features_list)
skip_start = len(subj_hdcat)

# (2) Fix erroneous values
data = fix_erroneous(data)

# (3) Select Baseline / Follow up only
data = data [data ['visit'].values ==  [visit_type]] # Selecting BASELINE visits only
data.drop_duplicates(subset="subjid", inplace=True, keep="first") # Removing duplicates

# (4) Get rid of unwanted cols - make rest numeric
data.drop(remove, axis=1, inplace=True) # > Unwanted columns set at beginning
isced = data["isced"] # Isolating covariate isced before dropping rest
data = data.drop(columns=data.columns[5:84]) # # Medical history / demographic features etc (leaving only test-related features)
data.insert(5, "isced", isced) # adding isced back in
# > Make all values numeric
for column in data.iloc [:, skip_start:]: # Except subject id, hd_cat
    data [column] = pd.to_numeric(data [column])
# > Remove cols with > 40% NaNs
# print("Columns with > 40% NaNs (which have been dropped):")
# for column in data: # -1 to exclude isced *** This excludes isced unless done just on Baseline visits)
#     if (data [column].isna().sum() / data.shape [0]) > 0.4:
#         # print(column)
#         data.drop(column, inplace=True, axis=1)

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

# # (6) Normality test
# data_noNan = missing_vals(data, print_miss_counts=True, method="mean")
# for column in data_noNan.iloc [:, skip_start:]: # skipping subjid, hdcat
#     title = column
#     column = data_noNan [column]
#     stat, p = scipy.stats.normaltest(column)
#     print(title, round(stat, 4), round(p,8))

# # (7) Feature analysis
# t_test_results = feature_selection(data.iloc [:, 1:], "hdcat", method="t_test", sign_lev=0.05) # 1 to skip subjID
# wilcoxon_results = feature_selection(data.iloc [:, 1:], "hdcat", method="wilc", sign_lev=0.05) # 1 to skip subjID
# mann_whitney_results = feature_selection(data.iloc [:, 1:], "hdcat", method="mann_w", sign_lev=0.05)
# rf_importances = feature_selection_rf(data.iloc [:, 1:], "hdcat")

# (8) Features subset w covars (plus subjID and hdcat)
data_subj_hdcat = data [subj_hdcat]
data_features = data [features_list]
data_covariates = data [covariates_list]
data = pd.concat( [data_subj_hdcat, data_features, data_covariates], axis=1)

# (9) Deal with missing values
data = missing_vals(data, method="drop")

# (10) Remove outliers
# > Remove outliers 5*SDs from mean   *** NOTE: removes almost all mmse rows
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

# features_list = features_list [:-1] # removing hdcat from features list (as comes out of outlier_removal() with hdcat)

# (11) Data counts for write-up
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

# (12) Covariate adjustment(+ Z-scoring if sustainType=ZScoreSustain)
# Regresses out covariates of predictors + if ZScoreSustain; (2) Finds mean and SD of features, (3) Takes Z scores, (4) Makes sure all Z scores are increasing
data = covariate_adjustment(data, covariates_list, features_list, sustainType)
# Check adjustment has worked - i.e., mean of each control biomarker == 0.
feat = 0
for column in data [data [:, len(features_list)] == 0].T:  # PROBLEM   8 = hdcat. Check mean of each control biomarker now == 0 (means adjustment has worked)
    mean = np.mean(column)
    print("\n control mean, biomarker", feat, "=", np.around(mean, 4))
    feat = feat + 1

# ------------------------end of pre-processing-------------------------

# (13) RUN model using preprocessed data
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

        # Filling dataframes L_yes and L_no (each the shape 'data') with a probability value for each cell
        for i in range(N): # For each biomarker
            if sustainType == "mixture_GMM":
                L_no [:, i], L_yes [:, i] = mixtures[i].pdf(None, data [:, i]) # pdf() = function in kde_ebm. Returns controls_score, patholog_score (probability values)
            elif sustainType == "mixture_KDE":
                L_no [:, i], L_yes [:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1)) # CALCULATING PDF

        # Creating MixtureSustain() object
        sustain = MixtureSustain(L_yes, L_no, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder,
                                 dataset_name, use_parallel_startpoints=True)

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
    L_yes, L_no, log_likelihoods, ml_subtype, ml_stage, sustain, samples_f, samples_sequence, mixtures = main(data, sustainType)


## (14) T-test differences between subtypes

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


#### (15) T-tests between each subtype’s CAG distribution

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