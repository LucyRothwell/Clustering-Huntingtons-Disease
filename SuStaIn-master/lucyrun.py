###
# Adapted from:
# pySuStaIn: SuStaIn algorithm in Python (https://www.nature.com/articles/s41468-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
# Contributors: Leon Aksman (l.aksman@ucl.ac.uk), Arman Eshaghi (a.eshaghi@ucl.ac.uk), Alex Young (alexandra.young@kcl.ac.uk)
#######################################################################################################################

# CONTENTS
# (0) Define settings: Sustain type & features etc
# (1) Fix erroneous values
# (2) Select Baseline / Follow up only
# (3) Get rid of medical and demographic features (around 80 columns) + drop columns with >50% NaNs
# (4) Normality test
# (5) Feature selection
# (6) Features subset
# (7) Deal with missing values
# (8) Remove outliers
# (9) Covariate adjustment(+ Z-scoring if sustainType=ZScoreSustain)
# (10) RUN model using preprocessed data
# (11) T-test differences between subtypes

#######################################################################################################################
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from kde_ebm.mixture_model import fit_all_kde_models, fit_all_gmm_models
from kde_ebm import plotting
from matplotlib import cbook as cbook
import warnings
warnings.filterwarnings("ignore", category=cbook.mplDeprecation)
from ZscoreSustain import ZscoreSustain
from MixtureSustain import MixtureSustain
import sklearn.model_selection

from pkge0.preprocessing_pipeline import * # Created by author, Lucy Rothwell
from pkge0.feature_selection import * # Created by author, Lucy Rothwell

# PIPELINE: LOAD AND PRE-PROCESS DATA
enroll_df = pd.read_csv("/Users/lucyrothwell/Google_Drive/MSc_Comp/9_Dissertation HD/Data - Enroll HD/enroll.csv", delimiter=',')
profile_df = pd.read_csv("/Users/lucyrothwell/Google_Drive/MSc_Comp/9_Dissertation HD/Data - Enroll HD/profile.csv", delimiter=',')
data = re_code_hdcat(enroll_df) # Recoding the HD category column

# # ------------------ Histogram to check spread ----------------------------
# testing_var = "chorlue"
# df = enroll_df[enroll_df[testing_var] > -50]
# # df = df[df["hdcat"]==0]
# df = df[df[testing_var] < 150]
# tms = df[testing_var].values
# hist = []
# labels = []
# for i in np.unique(df['hdcat'].values):
#     if np.isnan(tms[df['hdcat'].values==i]).all():
#         continue
#     hist.append(tms[df['hdcat'].values==i])
#     labels.append(str(i))
# plt.hist(hist,label=labels,stacked=True)
# plt.yscale("log")
# plt.xlabel(testing_var)
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()
# # ------------------------------------------------------------------------

## (0) Define settings: Sustain type & features etc
sustainType = "mixture_KDE" # for use in functions later
num_covars = 2
features_list = ["subjid", "motscore", "chorrue", "sacinith", "sdmt2", "verfct7", "trlb3", "depscore", "irascore", "pbas3sv", "age", "isced", "hdcat"] # 9 fits - TEST - USING FS VARS
num_predictors = len(features_list)-num_covars-2 # 2 to also take away hdcat and subjid

# (1) Fix erroneous values
data = fix_erroneous(data)

# (2) Select Baseline / Follow up only
data = data[data['visit'].values == ["Baseline"]] # Selecting BASELINE/FU visits only
data.drop_duplicates(subset="subjid", inplace=True, keep="first") # Remocing duplicates

# (3) Get rid of medical and demographic features (around 80 columns) + drop columns with >50% NaNs
print("data(pre delete_med_dem).shape:", data.shape)
data = delete_med_dem(data)
print("data(post delete_med_dem).shape:", data.shape)
# > Make all values numeric
for column in data.iloc[:, 1:]: # Except subject id - col 0
    data[column] = pd.to_numeric(data[column])

print("Columns with > 40% NaNs (which have been dropped):")
for column in data.iloc[:, :-1]: # -1 to exclude isced
    if (data[column].isna().sum() / data.shape[0]) > 0.4:
        print(column)
        data.drop(column, inplace=True, axis=1)

# # (4) Normality test
# data_noNan = missing_vals(data, print_miss_counts=True, method="mean")
# for column in data_noNan.iloc[:, 1:]: # skipping subjid
#     title = column
#     column = data_noNan[column]
#     stat, p = scipy.stats.normaltest(column)
#     if stat < 300:
#         print(title, round(stat, 4), round(p,8))

# # (5) Feature selection
wilcoxon_results = feature_selection_wilcoxon(data.iloc[:, 1:])
# rf_importances = feature_selection_rf(data.iloc[:, 1:])
# ttest_results = feature_selection_ttest(data)

# (6) Features subset
data = data[features_list]
# data[data["hdcat"]==1].hist()

# (7) Deal with missing values
data = missing_vals(data, print_miss_counts=True, method="drop")

# (8) Remove outliers
# > Remove outliers 5*SDs from mean
disease = data[data["hdcat"]==1]
controls = data[data["hdcat"]==0]
preman = data[data["hdcat"]==2]
if sustainType == 'mixture_GMM' or sustainType == "mixture_KDE": # Include hdcat (a few extra steps for this)
    feat = features_list[1:-3] # # 1 for subjid
    feat.append(features_list[-1]) # Adding hdcat
    print("\n","DISEASE")
    disease = outlier_removal(disease, feat, sd_num=5) # 1 to skip subject id
    print("\n","CONTROLS")
    controls = outlier_removal(controls, feat, sd_num=5)  # 1 to skip subject id
    print("\n","PREMAN")
    preman = outlier_removal(preman, feat, sd_num=5)  # 1 to skip subject id
    data = pd.concat([disease, controls, preman])
elif sustainType == 'zscore': # Exclude hdcat.
    data = outlier_removal(data, features_list[1:-3], sd_num=5) # 1 for subjID still in

dis_cont = data[data["hdcat"] != 2]


# # ---------(Counting analysis data for write-up)--------------
# Merge data to get data counts
data_m = pd.DataFrame(data)
data_merge = pd.merge(data_m, profile_df, on="subjid", how='left')
# print(data_merge["hdcat"].value_counts(normalize=True)*100)
# print(data_merge["sex"].value_counts(normalize=True)*100)
# print(data_merge["region"].value_counts(normalize=True)*100)
# print(data_merge["race"].value_counts(normalize=True)*100)

# Getting CAG array to concatenate to output later
data_merge_no_preman = data_merge[data_merge["hdcat"] != 2]
caghigh = pd.to_numeric(data_merge_no_preman["caghigh"])
caglow = pd.to_numeric(data_merge_no_preman["caglow"])
type(caghigh.iloc[0])
row = 0
for i in caghigh:
    if i == 1:
        caghigh.iloc[row] = caglow.iloc[row]
    row = row + 1
cag = caghigh
# ---------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------------------
# (9) Covariate adjustment(+ Z-scoring if sustainType=ZScoreSustain)
# Regresses out covariates of predictors + if ZScoreSustain; (2) Finds mean and SD of features, (3) Takes Z scores, (4) Makes sure all Z scores are increasing
data = covariate_adjustment(data.iloc[:, 1:], sustainType, num_predictors)
# Check adjustment has worked - i.e., mean of each control biomarker == 0.
feat = 0
for column in data[data[:, -1] == 0].T:  # Check mean of each control biomarker now == 0 (means adjustment has worked)
    mean = np.mean(column)
    print("control mean, biomarker", feat, "=", np.around(mean, 4))
    feat = feat + 1

# # **** Do this if not doing covariate adjustemnt- to get rid of covariates ****
# data = pd.DataFrame(data)
# hdcat = data.iloc[:, -1]
# predictors = data.iloc[:, 1:len(features_list)-num_covars-1] # 1 to exclude subjid, 2 covariates and hdcat
# data = pd.concat([predictors, hdcat], axis=1)
# data = np.array(data)
# ----------------------------------------------------------------------------------------------------------------------



#------------- HISTS CHECK ------------------------
# data_hists = pd.DataFrame(data)
# data_hists = data_hists.iloc[:, :-1]
# data_hists.columns = ["motscore", "chorrue", "sacinith", "sdmt2", "verfct7", "trlb3", "depscore", "irascore", "pbas3sv"]
#
# # data_hist[data_hist.iloc[:, -1] == 1].hist()
# # data_hist[data_hist.iloc[:, -1] == 0].hist()
# # print(motscore.value_counts())
# data_hists.hist()
# -------------------------------------------------



# ------------------------end of pre-processing-------------------------------------------------------------------------

# (10) RUN model using preprocessed data
def main(data, sustainType):
    # cross-validation
    validate = True
    N_folds = 10 # recommended 10

    if sustainType == "mixture_KDE" or "mixture_GMM":
        N = data.shape[1]-1  # number of biomarkers - if kdemm - hdcat is still in data so we need to -1 to get num of biomarkers
    if sustainType == "zscore":
        N = data.shape[1]  # number of biomarkers

    print("N (1) in main()=", N)

    M = data.shape[0]  # number of observations ( e.g. subjects )

    N_startpoints = 25 # Number of starting points to use when fitting the subtypes hierarchichally. I'd suggest using 25.
    N_S_max = 3 # The maximum number of subtypes to fit. I'd suggest starting with a lower number - maybe three - and then increasing that if you're getting a significantly better fit with the maximum number of subtypes. You can judge this roughly from the MCMC plot. To properly evaluate the optimal number of subtypes you need to run cross-validation.

    # N_iterations_MCMC = int(1e5)  # The number of iterations for the MCMC sampling of the uncertainty in the progression pattern. I'd recommend using 1x10^5 or 1x10^6.
    N_iterations_MCMC = 100  # Use this one to make programme quicker on test runs. CHANGE back toint(1e5) for final run.

    # either 'mixture_GMM' or 'mixture_KDE' or 'zscore'
    sustainType = sustainType

    assert sustainType in ("mixture_GMM", "mixture_KDE", "zscore"), "sustainType should be either mixture_GMM, mixture_KDE or zscore"

    dataset_name = 'enroll.csv'
    output_folder = dataset_name + '_' + sustainType

    Z_vals = np.array([[1, 2, 3]] * N)  # Z-scores for each biomarker
    Z_max = np.array([5] * N)  # maximum z-score

    if sustainType == 'mixture_GMM' or sustainType == "mixture_KDE":

        case_control_all = data[data[:, -1] != 2] # Removing pre-manifest
        labels_case_control = case_control_all[:, -1] # getting labels (-1 to remove hdcat)
        data_case_control = case_control_all[:, :-1] # getting predictors (-1 to remove hdcat)
        data = data_case_control
        print("data.shape",data.shape)
        print("labels_case_control.shape", labels_case_control.shape)

        N = data.shape[1]  #hdcat already removed so no need for-1
        print("N (2) in main()=", N)

        SuStaInLabels = []
        SuStaInStageLabels = []

        for i in features_list[1:num_predictors+1]:
            SuStaInLabels.append(i)

        print("SuStaInLabels", SuStaInLabels)

    # FIT THE BIOMARKER DATA - FINDING A "CLUSTER" FOR DISEASE AND A "CLUSTER" FOR CONTROLS
    # I.e., Estimate the parameters (mean and variance) of the Gaussian (GMM) or estimate the non-parametric (KDE)
        if sustainType == "mixture_GMM":
            mixtures = fit_all_gmm_models(data, labels_case_control.astype(int))
        elif sustainType == "mixture_KDE":
            # mixtures = fit_all_kde_models(data, labels.astype(int)) # PROBLEM: Crashed line 61 in utils.py
            mixtures = fit_all_kde_models(data, labels_case_control.astype(int)) # PROBLEM: Crashed line 61 in utils.py

        outDir = "/Users/lucyrothwell/Google_Drive/MSc_Comp/9_Dissertation HD/SuStaIn-master/"

        fig, ax = plotting.mixture_model_grid(data_case_control, labels_case_control, mixtures, SuStaInLabels)
        fig.show()
        fig.savefig(os.path.join(outDir, 'kde_fits.png')) #outDir should be output_folder

        L_yes = np.zeros(data.shape)
        L_no = np.zeros(data.shape)

        # Calculating data likelihood for each biomarker, N:
        for i in range(N):
        # BREAKING mixtures LIKELIHOODS INTO NORMAL AND ABNORMAL (based on likelihoods)
            if sustainType == "mixture_GMM":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
            elif sustainType == "mixture_KDE":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1)) # CALCULATING PDF
                # L_yes meeans likelihoods for abnormal
        sustain = MixtureSustain(L_yes, L_no, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder,
                                 dataset_name, use_parallel_startpoints=True)

    elif sustainType == 'zscore':


        sustain = ZscoreSustain(data, Z_vals, Z_max, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC,
                                output_folder, dataset_name, False)

    # RUNNING SUSTAIN ON OUTPUT OF KDEMM / GMM
    ml_subtype = sustain.run_sustain_algorithm()

    # CROSS VALIDATION
    if validate: # DO CROSS VALIDATION. A way of optimising for number of clusters.
        # Output here (cross_validated_sequences) will be messier. Check there is differences between 1 cluster, 2 clusters etc.
        test_idxs = []

        cv = sklearn.model_selection.StratifiedKFold(n_splits=N_folds, shuffle=True)
        # cv_it = cv.split(data, labels)
        cv_it = cv.split(data, labels_case_control.astype(int))

        for train, test in cv_it:
            test_idxs.append(test)
        test_idxs = np.array(test_idxs)

        CVIC, loglike_matrix, y_arr_test = sustain.cross_validate_sustain_model(test_idxs)

        # this part estimates cross-validated positional variance diagrams
        for i in range(N_S_max):
            sustain.combine_cross_validated_sequences(i + 1, N_folds)

    plt.show()
    return y_arr_test, ml_subtype


if __name__ == '__main__':
    np.random.seed(42)
    # main(data, sustainType)  # (Original. Replaced with line below).
    y_arr_test, ml_subtype = main(data, sustainType)

## (11) T-test differences between subtypes

# print(ttest_ind(y_arr_test[0], y_arr_test[1]))
# print(ttest_ind(y_arr_test[1], y_arr_test[2]))
#
t_test_12, p_val_12 = ttest_ind(y_arr_test[1], y_arr_test[2])
t_test_23, p_val_23 = ttest_ind(y_arr_test[1], y_arr_test[2])
#
# p_val_12 = round(int(p_val_12), 4)
# p_val_23 = round(int(p_val_23), 4)


## (12) T-tests between each subtype’s CAG distribution

ml_subtype = pd.DataFrame(ml_subtype)
ml_subtype_wCAG = pd.concat([ml_subtype, cag], axis=1)
# ml_subtype_wCAG = ml_subtype_wCAG.dropna()

# print(ml_subtype_wCAG.iloc[:, 0].value_counts(normalize=True)*100)

sub1_cag = ml_subtype_wCAG.iloc[:, 1][ml_subtype_wCAG.iloc[:, 0]==0]
sub2_cag = ml_subtype_wCAG.iloc[:, 1][ml_subtype_wCAG.iloc[:, 0]==1]
sub3_cag = ml_subtype_wCAG.iloc[:, 1][ml_subtype_wCAG.iloc[:, 0]==2]

print(ttest_ind(sub1_cag, sub2_cag, nan_policy="omit"))
print(ttest_ind(sub2_cag, sub3_cag, nan_policy="omit"))
print(ttest_ind(sub1_cag, sub3_cag, nan_policy="omit"))