import pandas as pd
import numpy as np
from sklearn import linear_model
import scipy.stats

#######################################################################################################################

# CONTENTS
# (1) Fix erroneous values &  Re-code hdcat
# (2) Missing values
# (3) Remove outliers function
# (4) Normality test
# (5) Regress out covariates

#######################################################################################################################


# --------------------------- (1) FIX ERRONEOUS VALUES / RE-CODE HDCAT -------------------------------------------------
# Specific to enroll

def fix_erroneous(df):
    # Replace <18 ("age") with 17 (making it numeric)
    df.replace("<18", 17, inplace=True)  # Replaces <18 in "age" with 17 (making it numeric)

    # Remove isceds > 6 (scale ends at 6)
    df ['isced'].value_counts() # Count values
    data = df.drop(df [df.isced > 6].index) # isced is scored 0-6

    return data

def re_code_hdcat(df):
    # Re-code hdcat values: 0=control, 1=manifest, 2=pre-manifest
    df ["hdcat"].replace(3, 1, inplace=True)
    df ["hdcat"].replace(4, 0, inplace=True)
    df ["hdcat"].replace(5, 0, inplace=True)
    return df


# ------------------------------ (2) MISSING VALUES --------------------------------------------------------------------

def missing_vals(df, method, print_miss_counts=True):
    if print_miss_counts == True: # This prints number of missing values in each feature
        pd.set_option('display.max_rows', None)
        print("Dropped values =", df.isnull().sum())
    if method == "drop": # Drop missing values
        data = df.dropna()
    elif method == "mean": # Replace missing value with col mean
        disease = df [df ["hdcat"]==1]
        control = df [df ["hdcat"]==0]
        pre_man = df [df ["hdcat"]==2]
        # print("disease.mean() =", disease.mean)
        disease = disease.fillna(disease.mean())
        control = control.fillna(control.mean())
        pre_man = pre_man.fillna(pre_man.mean())
        data = pd.concat( [disease, control, pre_man], axis=0)
    else:
        print("Error: method must be either 'mean' or 'drop'")
    return data


# -------------------------- (3) REMOVING OUTLIERS ---------------------------------------------------------------------
# > Counts outliers more than 5 (can be set) SDs above or below the mean

def outlier_removal(data, features_list, sd_num=5):
    for i in features_list:
        print(i)
        rows1 = data.shape [0] # For working out total outliers per column
        sd_col = data [i].std()
        mean_col = data [i].mean()
        print("sd =", sd_col)
        data.drop(data [data [i] > mean_col + (sd_num * sd_col)].index, inplace=True, axis=0)
        data.drop(data [data [i] < mean_col - (sd_num * sd_col)].index, inplace=True, axis=0)
        rows2 = data.shape [0] # For working out total outliers per column
        outlier_count = rows1-rows2
        print("outlier_count =", outlier_count)
    return data


# -----------------------------(4) NORMALITY TEST ----------------------------------------------------------------------

def normality_test(data):
    for column in data.iloc [:, 1:]: # skipping subjid
        title = column
        column = data [column]
        # print(column.head)
        # print("column.shape", column.shape)
        # print("column.mean", column.mean)
        stat, p = scipy.stats.normaltest(column)
        # if p <0.05:
        print(title, round(stat, 4), round(p,8))

# -------------------- (5) REGRESSING OUT COVARIATES FUNCTIONS ---------------------------------------------------------
# > Prepares return data:
# - If sustainType='mixture_GMM' or "mixture_KDE" - Adds labels column back on before returning (KDE uses labels col) - rerturns hdcat = 0,1,2 (controls, disease, pre-man)
# - If sustainType='ZScoreSustain' - (1) Calculates means and SDs (2) Gets Z scores (3) Adjusts biomarkers that decrease with disease progression - returns disease only (hdcat=1?)

# HOW IT WORKS
# Runs linear regression on the control data for each biomarker where:
    # y = a biomarker
    # X = the 4 covariates
    # lr.fit(X=covariates(controls), y=data(controls))
# OUTPUTS: The 4 regression coefficients (ONE for each covariate) in predicting each biomarker
# COEFFICIENTS TELL US: The extent to which age, isced etc can predict each biomarker
# These coefficients are then substracted from the didease data, leaving only the disease effect

# Code for this function by Dr Peter Wijeratne:
def nu_freq(X_reg, y_reg): # Get regression params for each biomarker
    lr = linear_model.LinearRegression()
    lr.fit(X_reg, y_reg)
    return lr.coef_, lr.intercept_

# Code for this function by Dr Peter Wijeratne:
def controls_residuals(X, y, covar): # y used only to separate controls from disease group
    controls_covar = covar [y == 0] # taking covariates where y are controls
    controls_X = X [y == 0] # taking predictor data where y are controls
    n_biomarkers = controls_X.shape [1] # Number of biomarkers = num of columns in predictor data
    regr_params = dict((x, None) for x in range(n_biomarkers)) # {0: None, 1: None, 2: None, 3: None, 4: None} - i.e., creates a key in a dict for each biomarker, with value None:
    for i in range(n_biomarkers): # Loop creates regression coefficients for each biomarker
        mask = ~np.isnan(controls_X [:, i]) & ~np.isnan(controls_covar).any(axis=1) # This and next line simply remove nans
        # applying isnan() to all rows in biomarker i |&| applying isnan() to all covariates. |-| tilde = returns complement
        X_reg, y_reg = controls_covar [mask, :], controls_X [mask, i] # ??? Applying(?) mask to the covariates and predictor data - putting in X_reg, y_reg
        # Could line below be done directly on controls_covar and controls_X if NaNs have already been removed?
        regr_params [i] = nu_freq(X_reg, y_reg) # Get regression coef and intercept for each biomarker and put them in values of dict
    residuals = X.copy() # Just creating a shallow copy of X
    for key, value in regr_params.items(): # TAKING THE COEFFICIENTS AWAY FROM EACH VALUE IN EACH BIOMARKER. For each biomarker in the regr_params dict
        residuals [:, key] -= regr_params [key] [1] # resids [] = resids [] - regr_params []
        residuals [:, key] -= np.matmul(covar, regr_params [key] [0]) # resids [] = resids [] - regr_params []
    # data = np.round((residuals), 4)  # Round all values to 4 decimal places
    return residuals # Produces adjusted residuals


def covariate_adjustment(data, covariates_list, features_list, sustainType, return_stats=False):
    # (a) Getting input data for controls_residuals()
    # array_of_covariate_data.shape = (Number of people, Number of covariates)
    array_of_covariate_data = np.array(data [covariates_list], dtype=np.float64) # # Selecting only the covariates
    print("array_of_covariate_data.shape", array_of_covariate_data.shape)
    # array_of_clinical_labels.shape = (Number of people)
    array_of_clinical_labels = np.array(data ["hdcat"]) # Selecting only the label column
    print("array_of_clinical_labels.shape", array_of_clinical_labels.shape)
    # array_of_data_to_be_adjusted.shape = (Number of people, Number of biomarkers)
    array_of_data_to_be_adjusted = np.array(data [features_list], dtype=np.float64) # Selecting only the predictor columns
    print("array_of_data_to_be_adjusted.shape", array_of_data_to_be_adjusted.shape)

    # (b) RUN controls residuals:
    data_adjusted = controls_residuals(array_of_data_to_be_adjusted, array_of_clinical_labels, array_of_covariate_data)

    # (c) Prepare data for return depending on sustainType
    if sustainType == 'mixture_GMM' or sustainType == "mixture_KDE": # Add labels column back on before returning (KDE uses labels col)
        data_adjusted = np.concatenate((data_adjusted, array_of_clinical_labels [:, None]), axis=1)
        return data_adjusted

    # vv *** NEEDS TESTED *** vv
    elif sustainType == "zscore": # Take z scores of data and return without labels column
        # Re-splitting updated data for step (2)
        array_of_clinical_labels = pd.DataFrame(array_of_clinical_labels)
        data_with_labels = pd.concat( [data_adjusted, array_of_clinical_labels], axis=1)
        data_with_labels.columns =  ["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "labels"]
        # data = data_with_labels [data_with_labels ['labels'].values ==  [1]].iloc [:, :5]
        data = data_with_labels [data_with_labels ['labels'].values ==  [1]].iloc [:, :len(features_list)]
        # data_controls = data_with_labels [data_with_labels ['labels'].values ==  [0]].iloc [:, :5] # Subset controls for use in z score modelling
        data_controls = data_with_labels [data_with_labels ['labels'].values ==  [0]].iloc [:, :features_list] # Subset controls for use in z score modelling

        # (2) Calculate the mean and standard deviation of each biomarker in your controls dataset
        mean_controls = np.mean(data_controls, axis=0)
        std_controls = np.std(data_controls, axis=0)

        # (3) Z-score your data by taking (data-mean_controls)/std_controls.
        data = (data - mean_controls) / std_controls
        data_controls = (data_controls - mean_controls) / std_controls

        # (4) Identify any biomarkers that decrease with disease progression, these will have mean_data < mean_controls.
        # Multiply the data for these biomarkers by -1.
        # *** NEEDS AUTOMATED
        IS_decreasing_1 = np.mean(data, axis=0) < np.mean(data_controls, axis=0)
        print(IS_decreasing_1)
        data = pd.DataFrame((data - mean_controls) / std_controls)
        data_controls = pd.DataFrame((data_controls - mean_controls) / std_controls)
        data.iloc [:, 0] = data.iloc [:, 0].mul(-1) # motscore
        data.iloc [:, 2] = data.iloc [:, 2].mul(-1) # mmsetotal
        data.iloc [:, 3] = data.iloc [:, 3].mul(-1) # irascore
        data.iloc [:, 4] = data.iloc [:, 4].mul(-1) # exfscore
        IS_decreasing_2 = np.mean(data, axis=0) < np.mean(data_controls, axis=0)
        print(IS_decreasing_2)
        if return_stats==False:
            return np.array(data)
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
            return np.array(data)


# -------------------- (6) POST HOC - PREDICTING STAGE & SUBTYPE OF INDIVIDUALS-----------------------------------------

def get_subtype_stage(new_patient_data, features_list, sustainType, mixtures, sustain, samples_sequence, samples_f, N_samples=1000):
    import numpy as np
    from MixtureSustain import MixtureSustain
    subjid = new_patient_data["subjid"]
    print("\n","\n","\n" "----- Patient", str(subjid), "-----")

    new_patient_data = np.asarray(new_patient_data.iloc[:, :-1]) # -1 to get rid of hdcat
    # print("new_patient_data[0, :]" "\n",new_patient_data[0, :]) # Checking we got the 1st visit
    # print("new_patient_data[-1, :]" "\n", new_patient_data[-1, :]) # Checking we got the final visit
    L_yes_new = np.zeros(new_patient_data.shape)  # ...the probability that each value in dataframe is DISEASE (across all biomarkers)
    L_no_new = np.zeros(new_patient_data.shape)  # ...the probability that each value in dataframe is HEALTHY (across all biomarkers)

    N = len(features_list)
    print("sustainType =", sustainType)
    print("type(new_patient_data)", type(new_patient_data))
    # Filling the L arrays with probabilities

    for i in range(1, N+1): # (for each biomarker)
        if sustainType == "mixture_KDE":
            # np.append(L_no_new, mixtures[i].pdf(new_patient_data[:, i]), axis=1), np.append(L_yes_new, mixtures[i].pdf(new_patient_data[:, i].reshape(-1, 1)), axis=1)  # Add .reshape(-1, 1) if using array  # CALCULATING PDF
            L_no_new[:, i], L_yes_new[:, i] = mixtures[i].pdf(new_patient_data[:, i].reshape(1,-1))  # Add .reshape(-1, 1) if using array  # CALCULATING PDF
        elif sustainType == "mixture_GMM":
            L_no_new[:, i], L_yes_new[:, i] = mixtures[i].pdf(None, new_patient_data[i])  # pdf() = function in kde_ebm. Returns controls_score, patholog_score (probability values)

    # Selecting first (0) and last (1) visits and reshaping to give 2 dimensions
    L_yes_new0, L_no_new0 = L_yes_new[0, :].reshape(1, -1), L_no_new[0, :].reshape(1, -1)
    L_yes_new1, L_no_new1 = L_yes_new[-1, :].reshape(1, -1), L_no_new[-1, :].reshape(1, -1) # first -1 - to take the last visit a person has (to create biggest time gap between measurements 1 and 2)

    # Getting subtype and stage of visit 1 and visit 2 of the patient
    ml_subtype_v1, prob_ml_subtype_v1, ml_stage_v1, prob_ml_stage_v1 = MixtureSustain.subtype_and_stage_individuals_newData \
        (sustain, L_yes_new0, L_no_new0, samples_sequence, samples_f, N_samples)
    print("\n" "Visit 1 (Baseline)" "\n"  "ml_subtype:", ml_subtype_v1, "\n" "prob_ml_subtype:", prob_ml_subtype_v1,
          "\n" "ml_stage:", ml_stage_v1, "\n" "prob_ml_stage:", prob_ml_stage_v1)

    ml_subtype_v2, prob_ml_subtype_v2, ml_stage_v2, prob_ml_stage_v2 = MixtureSustain.subtype_and_stage_individuals_newData \
        (sustain, L_yes_new1, L_no_new1, samples_sequence, samples_f, N_samples)
    print("\n" "Visit Latest" "\n" "ml_subtype:", ml_subtype_v2, "\n" "prob_ml_subtype:", prob_ml_subtype_v2,
          "\n" "ml_stage:", ml_stage_v2, "\n" "prob_ml_stage:", prob_ml_stage_v2)

    return ml_subtype_v1, prob_ml_subtype_v1, ml_stage_v1, prob_ml_stage_v1, ml_subtype_v2, prob_ml_subtype_v2, ml_stage_v2, prob_ml_stage_v2
