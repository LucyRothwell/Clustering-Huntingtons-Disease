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
df = pd.read_csv("/Users/lucyrothwell/Google_Drive/MSc_Comp/9_Dissertation HD/Data - Enroll HD/enroll.csv", delimiter=',')

# Define Sustain Type
sustainType = "mixture_KDE" # for use in functions later

# (0) Remove erroneous values specific to enroll
data = fix_erroneous(df)

#--------------- counting---------------------
# print(df["caghigh"])
# baseline = df[df["visit"] == "Baseline"]
# controls = baseline[baseline["hdcat"] == 0]
# disease = baseline[baseline["hdcat"] == 1]
# preman = baseline[baseline["hdcat"] == 2]
#---------------------------------------------

# (1) Feature selection
# ttest_results = feature_selection_ttest(data)
rf_importances = feature_selection_rf(data) # Gives totally different results from ttest...

# Choose 5 features (based on feature_selection()) + 2 covariates + hdcat
features_list = ["scnt1", "swrt1", "sit1", "mmsetotal", "brady",  "age", "isced", "hdcat"] # 4 fits - COGNITIVE ONLY ("total correct" on 5 tests)
# features_list = ["sit2", "mmsetotal", "tfcscore", "clb", "exfscore", "age", "isced", "hdcat"] # 2 fits
# features_list = ["pakfrq", "bar", "rigarml", "cocfrq", "herfrq",  "age", "isced", "hdcat"] # X fits - T-TEST  >>ALL ARE SUBSTANCES-BASED
# features_list = ["dystlue", "prosupr", "brady", "sacinitv", "dystrle",  "age", "isced", "hdcat"] # X fits - T-TEST symptom tests  >>ALL ARE MOTOR-BASED

# features_list = ["clb", "clbfrq", "hxbar", "hxtrq", "hxpakfrq",  "age", "isced", "hdcat"] # 0 FITS - RF TOP 5
# features_list = ["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "age", "isced", "hdcat"] # ORGINAL

# motscore (motor - Motor Diagnostic Confidence) - Motor test score. p=0.0
# tfcscore (motor? - Total Functional Capacity) - Total functional score. p=0.0
# ocularh (motor - Motor Diagnostic Confidence) - Group occular pursuit: horizontal. p=0.0
# sacinith (motor - Motor Diagnostic Confidence) - Group Saccade initiation: horizontal. p=0.0
# fingtapr (motor - Motor Diagnostic Confidence) - Finger tap right. p=0.0
# brady (motor - Motor Diagnostic Confidence) - Bradykinesia‐body. p=0.0
# dystlle (motor - Motor Diagnostic Confidence) - Group maximal dystonia. p=0.0002
# ---
# verfct6 (cogn - Cognitive Assessment) - Verbal fluency test: Total intrusion errors. p=0.0193
# sit2 (cogn - Cognitive Assessments) - Stroop Interference: total errors. p=0.0243
# swrt3 (cogn - Cognitive Assessments) - Stroop: Total self‐corrected errors. p=0.0249 (**SAME AS sit2**)
# scnt2 (cogn - Cognitive Assessments) - Stroop - total errors. p=0.0077
# ---
# pbas1sv (psych - Problem Behaviours Assessment) - Group depressed mood: Severity. p=0.0391.
# ---
# fascore (gen - UHDRS Functional Assessment Independence Scale) - Functional assmt score. p=0.0002
# indepscl (gen - UHDRS Functional Assessment Independence Scale) - Subject's independence in %. p=0.0
# p > 0.05: "mmsetotal", "irascore", "exfscore"
# ---
# GOOD P CURVE ON:
# "mmsetotal" (p-value > 0.05?)
# "sit2" (varies based on which other vars are in feature_list - presumably because different nas get dropped)
# "brady" (varies based on which other vars are in feature_list - presumably because different nas get dropped)

# (2) Subset data
# > Returns features specified in features_list (selected features + covariates + hdcat) and drops na
print("data(pre subset).shape:", df.shape)
data = subset(df, features_list)

# Make all values numeric
for column in data:
    data[column] = pd.to_numeric(data[column])

# AD HOC FIT FIXES
# data = data[data["brady"] < 5]
data = data[data["mmsetotal"] > -250] # WORKS
data = data[data["mmsetotal"] < 7]
# for column in data:
#     data = data[data[column] < 40]
# data = data[data["prosupr"] < 8] # WORKS
# data = data[data["sacinitv"] < 5]
# data = data[data["sdmt1"] < 2] # LOST CAUSE
print("data(post subset).shape:", data.shape) # Prints data normally


# (3) Covariate adjustment(+ Z-scoring if sustainType=ZScoreSustain)
# Regresses out covariates of predictors + if ZScoreSustain; (2) Finds mean and SD of features, (3) Takes Z scores, (4) Makes sure all Z scores are increasing
data = covariate_adjustment(data, sustainType)



# (4) Remove outliers
# > Count outliers
# > Remove outliers 5*SDs from mean (do before pre-processing so outliers don't affect covariate adjustment and Z-scores?)
if sustainType == 'mixture_GMM' or sustainType == "mixture_KDE": # Include hdcat
    feat = features_list[0:-3]
    feat.append(features_list[7])
    data = outlier_removal(data[:,0:6], feat, sd_num=5, remove=True)
elif sustainType == 'zscore': # Exclude hdcat
    data = outlier_removal(data[:, 0:5], features_list[0:-3], sd_num=5, remove=True)


# > Check dists
# data = pd.DataFrame(data) # Converting to df so can use .hist()
# data.hist(bins=100)
# data = np.array(data) # Changing back for rest of prog

# (5) RUN model using preprocessed data
def main(data, sustainType):
    # cross-validation
    validate = True
    N_folds = 10 # recommended 10

    N = df.shape[1]  # number of biomarkers
    M = df.shape[0]  # number of observations ( e.g. subjects )
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

        # data.columns = ["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "hdcat"]
        # print("data(2):", data)

        case_control_all = data[data[:, 5] != 2]
        print("case_control_all.shape", case_control_all.shape)
        labels_case_control = case_control_all[:, 5]
        print("labels_case_control.shape", labels_case_control.shape)
        # data_case_control = case_control_all[:, range(1,3)]
        data_case_control = case_control_all[:, :5]
        print("data_case_control.shape", data_case_control.shape)

        data = data_case_control

        ## 2 VARS
        # labels = data[:, 5]
        # data = data[:, 1:5]
        N = data.shape[1]
        # # data.hist()
        #
        SuStaInLabels = []
        SuStaInStageLabels = []
        # ['Biomarker 0', 'Biomarker 1', ..., 'Biomarker N' ]
        #
        for i in range(N):
            SuStaInLabels.append('Biomarker ' + str(i))

        # print("data(3)", data)
        if sustainType == "mixture_GMM":
            mixtures = fit_all_gmm_models(data, data[:, 5])
        elif sustainType == "mixture_KDE":
            # mixtures = fit_all_kde_models(data, labels.astype(int)) # PROBLEM: Crashed line 61 in utils.py
            mixtures = fit_all_kde_models(data, labels_case_control.astype(int)) # PROBLEM: Crashed line 61 in utils.py

        outDir = "/Users/lucyrothwell/Google_Drive/MSc_Comp/9_Dissertation HD/SuStaIn-master/"
        fig, ax = plotting.mixture_model_grid(data_case_control, labels_case_control, mixtures, SuStaInLabels)
        fig.show()
        fig.savefig(os.path.join(outDir, 'kde_fits.png')) #outDir should be output_folder

        L_yes = np.zeros(data.shape)
        L_no = np.zeros(data.shape)

        for i in range(N): # Calculating data likelihood for each biomarker, N
        # BREAKING mixtures LIKELIHOODS INTO NORMAL AND ABNORMAL (based on likelihoods)
        # for i in range(1,3):
            if sustainType == "mixture_GMM":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(None, data[:, i])
            elif sustainType == "mixture_KDE":
                L_no[:, i], L_yes[:, i] = mixtures[i].pdf(data[:, i].reshape(-1, 1)) # CALCULATING PDF
                # L_yes = likelihoods for abnormal

        # RUNNING SUSTAIN ON OUTPUT OF KDEMM
        sustain = MixtureSustain(L_yes, L_no, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC, output_folder,
                                 dataset_name, use_parallel_startpoints=True)

    elif sustainType == 'zscore':

        sustain = ZscoreSustain(data, Z_vals, Z_max, SuStaInLabels, N_startpoints, N_S_max, N_iterations_MCMC,
                                output_folder, dataset_name, False)

    sustain.run_sustain_algorithm() # PROBLEM_Crashes line 239 in MixtureSustain.py

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
