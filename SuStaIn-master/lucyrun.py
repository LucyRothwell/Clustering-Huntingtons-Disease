###
# pySuStaIn: SuStaIn algorithm in Python (https://www.nature.com/articles/s41468-018-05892-0)
# Author: Peter Wijeratne (p.wijeratne@ucl.ac.uk)
# Contributors: Leon Aksman (l.aksman@ucl.ac.uk), Arman Eshaghi (a.eshaghi@ucl.ac.uk), Alex Young (alexandra.young@kcl.ac.uk)
###
import numpy as np
import pandas as pd
from statsmodels.sandbox.tsa.try_arma_more import sd
from simfuncs import generate_random_sustain_model, generate_data_sustain

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
from pkge0.preprocessing_pipeline import *
from pkge0.feature_selection import *

# pd.set_option('display.max_rows', None) # Just a display setting for df printing
# pd.set_option('display.max_columns', None)

# PIPELINE: LOAD AND PRE-PROCESS DATA
enroll_df = pd.read_csv("/Users/lucyrothwell/Google_Drive/MSc_Comp/9_Dissertation HD/Data - Enroll HD/enroll.csv", delimiter=',')
profile_df = pd.read_csv("/Users/lucyrothwell/Google_Drive/MSc_Comp/9_Dissertation HD/Data - Enroll HD/profile.csv", delimiter=',')
# profile_df = profile_df[profile_df["visit"]=="Baseline"]
data = re_code_hdcat(enroll_df)

# # ------------------ Pete's hist (1/3)------------------------------------------
testing_var = "sit1"
# df = enroll_df[enroll_df["mmsetotal"] > -50]
# df = df[df["mmsetotal"] < 150]
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

# # --------------- gene_neg / fam_control exploration -------------------
# gene_neg = enroll_df[enroll_df["hdcat"]==4]
# fam_control = enroll_df[enroll_df["hdcat"]==5]
#
# gene_neg_base = gene_neg[gene_neg["visit"]=="Baseline"]
# fam_control_base = fam_control[fam_control["visit"]=="Baseline"]

# gene_neg_sub = gene_neg["subjid", "dystrle", "sacvelv", "dystlue", "rigarml", "brady", "age", "isced", "hdcat"]
# fam_control = fam_control["subjid", "dystrle", "sacvelv", "dystlue", "rigarml", "brady", "age", "isced", "hdcat"]
# -----------------------------------------------------------------------

#------------- HISTS ------------------------
# disease = enroll_df[enroll_df["hdcat"]==1]
# controls = enroll_df[enroll_df["hdcat"]==0]
# motscore = disease["motscore"]
# print(motscore.value_counts())
# -------------------------------------------

## (0) Define settings: Sustain type & features etc
sustainType = "mixture_KDE" # for use in functions later
num_covars = 2

# features_list = ["subjid", "motscore", "rigarml", "dystlue", "chorbol", "chorrue", "sacinith", "tfcscore", "verfct7", "swrt2", "age", "isced", "hdcat"] # 5 FITS -  3 motor, 1 cog + tfc score - (AD HOC ON tfcscore)
features_list = ["subjid", "motscore", "rigarml", "dystlue", "swrt2", "tfcscore", "age", "isced", "hdcat"] # 5 FITS -  3 motor, 1 cog + tfc score - (AD HOC ON tfcscore)
# features_list = ["subjid", "motscore", "dystlue", "rigarml", "chorrue", "tfcscore", "age", "isced", "hdcat"] # 5 FITS -  4 motor + tfc score - (AD HOC ON tfcscore)
# features_list = ["subjid", "motscore", "dystlue", "rigarml", "chorbol", "chorlue", "age", "isced", "hdcat"] # 5 FITS - ALL MOTOR - (NO AD HOC)

# features_list = ["subjid", "chorbol", "chorlue", "finances", "tfcscore", "diagconf", "age", "isced", "hdcat"] # Wilcoxon top 5
# features_list = ["subjid", "chorbol", "chorlue", "chorrue", "chortrnk", "chorrle", "age", "isced", "hdcat"] # Wilcoxon chor
# features_list = ["subjid", "brady", "dystrle", "rigarml", "sacvelh", "dystlue", "age", "isced", "hdcat"] # T-test top 5 tests
# features_list = ["subjid", "dystrle", "sacvelv", "dystlue", "rigarml", "brady", "age", "isced", "hdcat"] #  RF top 5 tests

num_predictors = len(features_list)-num_covars-2 # 2 to also take away hdcat and subjid


# (1) Fix erroneous values
data = fix_erroneous(data)

# (2) Select Baseline / Follow up only
data = data[data['visit'].values == ["Follow Up"]] # Selecting BASELINE/FU visits only
data.drop_duplicates(subset="subjid", inplace=True, keep="first")

# (3) Get rid of medical and demographic features (around 80 columns)
print("data(pre delete_med_dem).shape:", data.shape)
data = delete_med_dem(data)
print("data(post delete_med_dem).shape:", data.shape)

# Make all values numeric
for column in data.iloc[:, 1:]: # Except subject id - col 0
    data[column] = pd.to_numeric(data[column])


## (4) Normality test
# data_noNan = missing_vals(data, print_miss_counts=True, method="mean")
# for column in data_noNan.iloc[:, 1:]: # skipping subjid
#     title = column
#     column = data_noNan[column]
#     # print(column.head)
#     # print("column.shape", column.shape)
#     # print("column.mean", column.mean)
#     stat, p = scipy.stats.normaltest(column)
#     if stat < 300:
#         print(title, round(stat, 4), round(p,8))


# (5) Feature selection
# ttest_results = feature_selection_ttest(data)
# wilcoxon_results = feature_selection_wilcoxon(data)
# rf_importances = feature_selection_rf(data.iloc[:, 1:])


# # ------------------ Pete's hist (2/3) ------------------------------------------
# df = data
# tms = df[test_var].values
# hist = []
# labels = []
# for i in np.unique(df['hdcat'].values):
#     if np.isnan(tms[df['hdcat'].values==i]).all():
#         continue
#     hist.append(tms[df['hdcat'].values==i])
#     labels.append(str(i))
# plt.hist(hist,label=labels,stacked=True)
# plt.yscale("log")
# plt.xlabel(test_var)
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()
# # ------------------------------------------------------------------------


# (6) Features suset
data = data[features_list]
# data[data["hdcat"]==1].hist()

# (7) Deal with missing values
data = missing_vals(data, print_miss_counts=True, method="mean")
# NOTE: if we drop NaNs (1) dystlue loses fit, (2) we need to cap swrt2 (see outliers funct)

# -----------------------------------------------------------------

# (8) Remove outliers
# > Remove outliers 5*SDs from mean
# (NEED TO DO OUTLIER REMOVAL ON EACH SUBJECT GROUP SEPARATELY THEN CONCATENATE)
disease = data[data["hdcat"]==1]
controls = data[data["hdcat"]==0]
preman = data[data["hdcat"]==2]
if sustainType == 'mixture_GMM' or sustainType == "mixture_KDE": # Include hdcat (a few extra steps for this)
    feat = features_list[1:-3] # # 1 for subjid
    feat.append(features_list[-1]) # Adding hdcat
    # data_feat_cols = data.iloc[:, 0:6] # Getting features only (not covariates, subjid etc)
    # pd.concat([data_feat_cols, data.iloc[:, -1]]) # Adding hdcat
    print("\n","DISEASE")
    disease = outlier_removal(disease, feat, sd_num=5) # 1 to skip subject id
    print("\n","CONTROLS")
    controls = outlier_removal(controls, feat, sd_num=5)  # 1 to skip subject id
    print("\n","PREMAN")
    preman = outlier_removal(preman, feat, sd_num=5)  # 1 to skip subject id
    data = pd.concat([disease, controls, preman])

    data = data[data["tfcscore"] < 60] # NEEDED if drop or mean NaNs
    # data = data[data["swrt2"] < 80] # NEEDED if do drop NaNs
    # NOTE: dystlue loses fit if drop NaNs

elif sustainType == 'zscore': # Exclude hdcat.
    data = outlier_removal(data, features_list[1:-3], sd_num=5) # 1 for subjID still in

# ------------------ Pete's hist (3/3)------------------------------------------
# df = data
# tms = df[test_var].values
# hist = []
# labels = []
# for i in np.unique(df['hdcat'].values):
#     if np.isnan(tms[df['hdcat'].values==i]).all():
#         continue
#     hist.append(tms[df['hdcat'].values==i])
#     labels.append(str(i))
# plt.hist(hist,label=labels,stacked=True)
# plt.yscale("log")
# plt.xlabel(test_var)
# plt.ylabel('Frequency')
# plt.legend()
# plt.show()
# # # ------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# (9) Covariate adjustment(+ Z-scoring if sustainType=ZScoreSustain)
# Regresses out covariates of predictors + if ZScoreSustain; (2) Finds mean and SD of features, (3) Takes Z scores, (4) Makes sure all Z scores are increasing
# data = covariate_adjustment(data, sustainType, num_predictors)
# # Check adjustment has worked - i.e., mean of each control biomarker == 0.
# feat = 0
# for column in data[data[:, -1] == 0].T:  # Check mean of each control biomarker now == 0 (means adjustment has worked)
#     mean = np.mean(column)
#     print("control mean, biomarker", feat, "=", np.around(mean, 4))
#     feat = feat + 1

# **** Do this if not doing covariate adjustemnt- to get rid of covariates ****
data = pd.DataFrame(data)
hdcat = data.iloc[:, -1]
predictors = data.iloc[:, 1:len(features_list)-num_covars-1] # 1 to exclude subjid, 2 covariates and hdcat
data = pd.concat([predictors, hdcat], axis=1)
data = np.array(data)
# ----------------------------------------------------------------------------------------------------------------------


#------------- HISTS CHECK ------------------------
# data_hist = pd.DataFrame(data)
# data_hist[data_hist.iloc[:, -1] == 1].hist()
# data_hist[data_hist.iloc[:, -1] == 0].hist()
# print(motscore.value_counts())
# -------------------------------------------------

# ---------(Counting analysis data for write-up)------------
# Merge data to get data counts
# data_merge = pd.merge(data, profile_df, on="subjid", how='left')
# print(data_merge["hdcat"].value_counts(normalize=True)*100)
# print(data_merge["sex"].value_counts(normalize=True)*100)
# print(data_merge["region"].value_counts(normalize=True)*100)
# print(data_merge["race"].value_counts(normalize=True)*100)
# print(data_merge["caghigh"].value_counts(normalize=True)*100)
# print(data_merge["caglow"].value_counts(normalize=True)*100)
# # Num. baseline visits in total enroll_hd db
# baseline = enroll_df[enroll_df["visit"]=="Baseline"]
# print(baseline)
# # Get rid of "subjid" column (no longer needed - was only needed for merge)
# data = data.iloc[:, 1:]
# # Getting rid of subjID column - no longer needed
# data = data.iloc[:,1:]
# ----------------------------------------------------


# ------------------------end of pre-processing-------------------------------------------------------------------------

# (8) RUN model using preprocessed data
def main(data, sustainType):
    # cross-validation
    validate = True
    N_folds = 10 # recommended 10

    if sustainType == "mixture_KDE" or "mixture_GMM":
        N = data.shape[1]-1  # number of biomarkers - if kdemm - hdcat is still in data so we need to -1 to get num of biomarkers
    if sustainType == "zscore":
        N = data.shape[1]  # number of biomarkers

    M = data.shape[0]  # number of observations ( e.g. subjects )
    # N_S_gt = 3  # number of ground truth subtypes

    N_startpoints = 25 # Number of starting points to use when fitting the subtypes hierarchichally. I'd suggest using 25.
    N_S_max = 3 # The maximum number of subtypes to fit. I'd suggest starting with a lower number - maybe three - and then increasing that if you're getting a significantly better fit with the maximum number of subtypes. You can judge this roughly from the MCMC plot. To properly evaluate the optimal number of subtypes you need to run cross-validation.
    # v CHANGE BACK v
    # N_iterations_MCMC = int(1e5)  # The number of iterations for the MCMC sampling of the uncertainty in the progression pattern. I'd recommend using 1x10^5 or 1x10^6.
    N_iterations_MCMC = 100  # The number of iterations for the MCMC sampling of the uncertainty in the progression pattern. I'd recommend using 1x10^5 or 1x10^6.

    # either 'mixture_GMM' or 'mixture_KDE' or 'zscore'
    sustainType = sustainType

    assert sustainType in ("mixture_GMM", "mixture_KDE", "zscore"), "sustainType should be either mixture_GMM, mixture_KDE or zscore"

    dataset_name = 'enroll.csv'
    output_folder = dataset_name + '_' + sustainType #"/Users/lucyrothwell/Google_Drive/MSc_Comp/9_Dissertation HD/SuStaIn-master" # Choose an output folder for the results.

    Z_vals = np.array([[1, 2, 3]] * N)  # Z-scores for each biomarker
    Z_max = np.array([5] * N)  # maximum z-score

    # SuStaInLabels = []
    # SuStaInStageLabels = []
    # # ['Biomarker 0', 'Biomarker 1', ..., 'Biomarker N' ]
    #
    # for i in range(N):
    #     SuStaInLabels.append('Biomarker ' + str(i))

    # gt_f = [1 + 0.5 * x for x in range(N_S_gt)]
    # gt_f = [x / sum(gt_f) for x in gt_f][::-1]

    # # ground truth sequence for each subtype
    # gt_sequence = generate_random_sustain_model(Z_vals, N_S_gt)
    #
    # N_k_gt = np.sum(Z_vals > 0) + 1
    # subtypes = np.random.choice(range(N_S_gt), M, replace=True, p=gt_f)
    # stages = np.ceil(np.random.rand(M, 1) * (N_k_gt + 1)) - 1
    #
    # data, data_denoised, stage_value = generate_data_sustain(subtypes,
    #                                                          stages,
    #                                                          gt_sequence,
    #                                                          Z_vals,
    #                                                          Z_max)

      # Works
    # choose which subjects will be cases and which will be controls
    # index_case = np.where(data["hdcat"] == 1) # gets the row numbers of the data where hdcat=1 (cases)
    # index_control = np.where(data["hdcat"] == 0) # gets the row numbers of the data where hdcat=0 (controls)
    #
    # # print("index_control.shape =", index_control.shape)
    # # print("index_case.shape =", index_case.shape)
    # print("index_case =", index_case)
    # print("index_control =", index_control) # All correct
    #
    # # PROBLEM
    # labels = 2 * np.ones(data.shape[0], dtype=int)  # 2 = MCI, default assignment here
    # labels[index_case] = 0
    # labels[index_control] = 1

    # print("labels =", labels)
    # print("labels[index_case] =", labels[index_case])
    # print("labels[index_control] =", labels[index_control])



    if sustainType == 'mixture_GMM' or sustainType == "mixture_KDE":

        case_control_all = data[data[:, -1] != 2]
        labels_case_control = case_control_all[:, -1]
        data_case_control = case_control_all[:, :-1]
        data = data_case_control  # Changing back to np
        print("data.shape",data.shape)
        print("labels_case_control.shape", labels_case_control.shape)

        # N = data.shape[1]

        SuStaInLabels = []
        SuStaInStageLabels = []

        # ['Biomarker 0', 'Biomarker 1', ..., 'Biomarker N' ]
        # for i in range(N):
        #     SuStaInLabels.append('Biomarker ' + str(i))

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
        fig, ax = plotting.mixture_model_grid(data, labels_case_control, mixtures, SuStaInLabels)
        fig.show()
        fig.savefig(os.path.join(outDir, 'kde_fits.png')) #outDir should be output_folder

        L_yes = np.zeros(data.shape) # CHANGE? Why both .zeros?
        L_no = np.zeros(data.shape)

        # Calculating data likelihood for each biomarker, N:
        for i in range(N):
        # BREAKING mixtures LIKELIHOODS INTO NORMAL AND ABNORMAL (based on likelihoods)
        # for i in range(1,3):
            if sustainType == "mixture_GMM":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
            elif sustainType == "mixture_KDE":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1)) # CALCULATING PDF
                # L_yes meeans likelihoods for abnormal

        print("L_yes.shape", L_yes.shape)
        print("L_no.shape", L_no.shape)
        print("N_startpoints", N_startpoints)
        print("N_S_max", N_S_max)
        sustain = MixtureSustain(L_yes, L_no, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder,
                                 dataset_name, use_parallel_startpoints=True)

    elif sustainType == 'zscore':


        sustain = ZscoreSustain(data, Z_vals, Z_max, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC,
                                output_folder, dataset_name, False)

    # RUNNING SUSTAIN ON OUTPUT OF KDEMM / GMM
    sustain.run_sustain_algorithm() # PROBLEM_Crashes line 239 in MixtureSustain.py

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

        CVIC, loglike_matrix = sustain.cross_validate_sustain_model(test_idxs)

        # this part estimates cross-validated positional variance diagrams
        for i in range(N_S_max):
            sustain.combine_cross_validated_sequences(i + 1, N_folds)

    plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    main(data, sustainType)
