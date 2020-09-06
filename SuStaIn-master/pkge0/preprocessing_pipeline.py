# import the python packages needed to generate simulated data for the tutorial
import pandas as pd
import numpy as np
from sklearn import linear_model
import scipy.stats

# Contents
# (1) Re-code / Fix erroneous values
# (2) Subset data
# (3) Remove outliers
# (4) Regress out covariates

# --------------------------- (1) FIX ERRONEOUS VALUES / RE-CODE HDCAT -------------------------------------------------
# Specific to enroll

def fix_erroneous(df):
    # Replace <18 ("age") with 17 (making it numeric)
    df.replace("<18", 17, inplace=True)  # Replaces <18 in "age" with 17 (making it numeric)

    # Remove isceds > 6 (scale ends at 6)
    df['isced'].value_counts() # Count values
    data = df.drop(df[df.isced > 6].index) # isced is scored 0-6
    # print("data_baseline_290(>6 dropped).shape (returned from subset()):", df.shape)

    return data

def re_code_hdcat(df):
    # Re-code hdcat values: 0=control, 1=manifest, 2=pre-manifest
    df["hdcat"].replace(3, 1, inplace=True)
    df["hdcat"].replace(4, 0, inplace=True)
    df["hdcat"].replace(5, 0, inplace=True)
    return df
    # df.drop(df.loc[df['hdcat'] == 5].index, inplace=True) # Removing family controls
    # print(df["hdcat"].value_counts())


# -------------- (2) DELETE MEDICAL HISTORY AND DEMOGRTAPHIIC FEATURES (TO GET CLINICAL TESTS ONLY) --------------------
# + isced, age and hdcat

def delete_med_dem(df): # Removes all demographic, medical history and drug abuse variables (for feature selection)
    final_col = df.columns.get_loc("updhdh")
    columns = list(df.iloc[:, 1:final_col+1].columns)
    hdcat, age, isced = df["hdcat"], df["age"], df["isced"]
    # print("columns to drop =", columns)
    df.drop(axis=1, labels=columns, inplace=True)
    data = pd.concat([df, hdcat, age, isced], axis=1)
    return data  # Returns only clinical test features


# ------------------------------ (3) MISSING VALUES --------------------------------------------------------------------

def missing_vals(df, print_miss_counts, method):
    if print_miss_counts == True: # This prints number of missing values in each feature
        pd.set_option('display.max_rows', None)
        # print(df.isnull().sum())
    if method == "drop": # Drop missing values
        data = df.dropna()
    elif method == "mean": # Replace missing value with col mean
        disease = df[df["hdcat"]==1]
        control = df[df["hdcat"]==0]
        pre_man = df[df["hdcat"]==2]
        # print("disease.mean() =", disease.mean)
        disease = disease.fillna(disease.mean())
        control = control.fillna(control.mean())
        pre_man = pre_man.fillna(pre_man.mean())
        data = pd.concat([disease, control, pre_man], axis=0)
    else:
        print("method must be either 'mean' or 'drop'")
    return data


# -------------------------- (4) REMOVING OUTLIERS -------------------------
# > Counts ouliers more than 5 (can be set) SDs above or below the mean

def outlier_removal(data, features_list, sd_num=5, remove=False):
    for i in features_list[0:-1]:
        print(i)
        rows1 = data.shape[0] # For working out total outliers per column
        sd_col = data[i].std()
        print("sd =", sd_col)
        data.drop(data[data[i] > sd_num * sd_col].index, inplace=True, axis=0)
        data.drop(data[data[i] < -sd_num * sd_col].index, inplace=True, axis=0)
        rows2 = data.shape[0] # For working out total outliers per column
        outlier_count = rows1-rows2
        print("outlier_count =", outlier_count)
    return data


# -----------------------------NORMALITY TEST --------------------------------------------------------------------------

# def normality_test(data):
#     for column in data.iloc[:, 1:]: # skipping subjid
#         title = column
#         column = data[column]
#         # print(column.head)
#         # print("column.shape", column.shape)
#         # print("column.mean", column.mean)
#         stat, p = scipy.stats.normaltest(column)
#         # if p <0.05:
#         print(title, round(stat, 4), round(p,8))

# -------------------- (5) REGRESSING OUT COVARIATES FUNCTIONS ---------------------------------------------------------
# > Regresses out covariates
# > Prepares return data:
#   - If sustainType='mixture_GMM' or "mixture_KDE" - Adds labels column back on before returning (KDE uses labels col) - rerturns hdcat = 0,1,2 (controls, disease, pre-man)
#   - If sustainType='ZScoreSustain' - (1) Calculates means and SDs (2) Gets Z scores (3) Adjusts biomarkers that decrease with disease progression - returns disease only (hdcat=1?)

def nu_freq(X_reg, y_reg): # Get regression params for each biomarker
    lr = linear_model.LinearRegression()
    lr.fit(X_reg, y_reg)
    return lr.coef_, lr.intercept_

def controls_residuals(X, y, covar): # y used only to separate controls from disease group
    controls_covar = covar[y == 0] # taking covariates where y are controls
    controls_X = X[y == 0] # taking predictor data where y are controls
    n_biomarkers = controls_X.shape[1] # Number of biomarkers = num of columns in predictor data
    regr_params = dict((x, None) for x in range(n_biomarkers)) # {0: None, 1: None, 2: None, 3: None, 4: None} - i.e., creates a key in a dict for each biomarker, with value None:
    for i in range(n_biomarkers): # Loop creates regression coef for each biomarker
        mask = ~np.isnan(controls_X[:, i]) & ~np.isnan(controls_covar).any(axis=1) # ??? *** PROBLEM. This and next line simply remove nans
        # applying isnan() to all rows in biomarker i |&| applying isnan() to all covariates. |-| tilde = returns complement
        X_reg, y_reg = controls_covar[mask, :], controls_X[mask, i] # ??? Applying(?) mask to the covariates and predictor data - puttig in X_reg, y_reg
        # Could line below be done directly on controls_covar and controls_X if NaNs have already been removed?
        regr_params[i] = nu_freq(X_reg, y_reg) # Get regression coef and intercept for each biomarker and put them in values of dict
    residuals = X.copy() # Just creating a shallow copy of X
    for key, value in regr_params.items(): # TAKING THE COEFFICIENTS AWAY FROM EACH VALUE IN EACH BIOMARKER. For each biomarker in the regr_params dict
        residuals[:, key] -= regr_params[key][1] # resids[] = resids[] - regr_params[]
        residuals[:, key] -= np.matmul(covar, regr_params[key][0]) # resids[] = resids[] - regr_params[]
    # data = np.round((residuals), 4)  # Round all values to 4 decimal places
    return residuals # Produces adjusted residuals


def covariate_adjustment(data_w_covariates, sustainType, num_predictors, return_stats=False):
    # (a) Getting input data for controls_residuals()
    array_of_covariate_data = np.array(data_w_covariates[["age", "isced"]], dtype=np.float64) # # Selecting only the covariates
    array_of_clinical_labels = np.array(data_w_covariates["hdcat"]) # Selecting only the label column
    array_of_data_to_be_adjusted = np.array(data_w_covariates.iloc[:, 0:num_predictors], dtype=np.float64) # Selecting only the predictor columns
        # swap in arrays for your data; shapes of each array as follows:
        # array_of_data_to_be_adjusted.shape = (Number of people, Number of biomarkers)
        # array_of_clinical_labels.shape = (Number of people)
        # array_of_covariate_data.shape = (Number of people, Number of covariates)

    # (b) RUN controls residuals:
    # data_adjusted = pd.DataFrame(controls_residuals(array_of_data_to_be_adjusted, array_of_clinical_labels, array_of_covariate_data))
    data_adjusted = controls_residuals(array_of_data_to_be_adjusted, array_of_clinical_labels, array_of_covariate_data)

    # (c) Prepare data for return depending on sustainType
    # global data_preprocessed  # Making global so can be returned to main
    if sustainType == 'mixture_GMM' or sustainType == "mixture_KDE": # Add labels column back on before returning (KDE uses labels col)
        data_preprocessed = np.concatenate((data_adjusted, array_of_clinical_labels[:, None]), axis=1)
        return data_preprocessed

    # CHECK THIS elif WORKS
    elif sustainType == "zscore": # Take z scores of data and return without labels column
        # Re-splitting updated data for step (2)
        array_of_clinical_labels = pd.DataFrame(array_of_clinical_labels)
        data_with_labels = pd.concat([data_adjusted, array_of_clinical_labels], axis=1)
        data_with_labels.columns = ["motscore", "tfcscore", "mmsetotal", "irascore", "exfscore", "labels"]
        # data = data_with_labels[data_with_labels['labels'].values == [1]].iloc[:, :5]
        data = data_with_labels[data_with_labels['labels'].values == [1]].iloc[:, :num_predictors]
        # data_controls = data_with_labels[data_with_labels['labels'].values == [0]].iloc[:, :5] # Subset controls for use in z score modelling
        data_controls = data_with_labels[data_with_labels['labels'].values == [0]].iloc[:, :num_predictors] # Subset controls for use in z score modelling

        # (2) Calculate the mean and standard deviation of each biomarker in your controls dataset
        mean_controls = np.mean(data_controls, axis=0)
        std_controls = np.std(data_controls, axis=0)

        # (3) Z-score your data by taking (data-mean_controls)/std_controls.
        # data_zscore = scipy.stats.zscore(data) #  Gets different results...
        data = (data - mean_controls) / std_controls
        # print("data:", data.shape)
        data_controls = (data_controls - mean_controls) / std_controls

        # (4) Identify any biomarkers that decrease with disease progression, these will have mean_data < mean_controls.
        # Multiply the data for these biomarkers by -1.
        # *** NEEDS AUTOMATED
        IS_decreasing_1 = np.mean(data, axis=0) < np.mean(data_controls, axis=0)
        print(IS_decreasing_1)
        data = pd.DataFrame((data - mean_controls) / std_controls)
        data_controls = pd.DataFrame((data_controls - mean_controls) / std_controls)
        data.iloc[:, 0] = data.iloc[:, 0].mul(-1) # motscore
        data.iloc[:, 2] = data.iloc[:, 2].mul(-1) # mmsetotal
        data.iloc[:, 3] = data.iloc[:, 3].mul(-1) # irascore
        data.iloc[:, 4] = data.iloc[:, 4].mul(-1) # exfscore
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

