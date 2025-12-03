import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from itertools import zip_longest
from xgboost import XGBClassifier

from evaluation import *


def get_features(rec_num, all_features, types):
    """
    Extracts feature and label data for a list of record numbers.

    Parameters:
    ----------
    rec_num : list or array-like
        List of record IDs to select data for.
    all_features : DataFrame
        DataFrame containing all features.
    types : DataFrame
        DataFrame containing metadata (e.g., 'Subject' and 'type' columns).

    Returns:
    -------
    XX : DataFrame
        Concatenated feature data for selected records.
    yy : ndarray
        Corresponding labels (e.g., class/type).
    """
    
    # Select the feature rows where Patient_id matches in rec_num
    sel_feats = all_features.loc[all_features['Patient_id'].isin(rec_num)]
    # Arrange rows by ID
    sel_feats = sel_feats.sort_values('Sample_id')
    # Delete the Patient_id column
    X = sel_feats.drop(columns=['Patient_id', 'Sample_id'], errors='ignore')
    
    sel_labels = []
    for i in range(len(sel_feats)):
        pid = sel_feats.iloc[i]["Patient_id"]
        match = types.loc[types["Patient_id"] == pid, "Class"]
        sel_labels.append(match.iloc[0])
    
    y = np.array(sel_labels)
    
    return X, y


def get_model(sel_IDs, test_size, all_features, types, standardize, iter_num, top_feat, sel_type, model_name):
    """
    Trains and evaluates a logistic regression model with Recursive Feature Elimination (RFE)
    over multiple iterations using group-based splitting.

    Parameters:
    ----------
    sel_IDs : pd.DataFrame
        A DataFrame with two columns containing patient/sample IDs for each class (e.g., Apnea vs Hypopnea).
    test_size : float
        Proportion of the dataset to include in the test split (e.g., 0.2 for 20% test data).
    all_features : pd.DataFrame
        Feature matrix where rows are samples and columns are extracted features.
    types : pd.DataFrame
        Metadata corresponding to the samples, used to map patient/sample types.
    standardize : int
        Whether to apply standardization (1 = Yes, 0 = No).
    iter_num : int
        Number of iterations for training/testing splits.
    top_feat : int
        Number of top features to select using RFE.
    sel_type : list
        List of type labels (e.g., ['Lat_Apn', 'Lat_Hyp']) to be used in feature extraction.

    Returns:
    -------
    dct_MODEL : dict
        A dictionary containing predictions, probabilities, splits, and models for each iteration.
        Keys include:
            - 'dct_win_pred'
            - 'dct_win_proba'
            - 'dct_win_Xtrn'
            - 'dct_win_Xtst'
            - 'dct_win_ytrn'
            - 'dct_win_ytst'
            - 'df_win_stype'
            - 'models'
    feat_num : pd.DataFrame
        A DataFrame of selected features and their frequency across all iterations.

    Notes:
    -----
    The function assumes a separate helper function `get_features()` is defined elsewhere,
    which retrieves the correct rows/features for the given patients and type labels.
    """
    # Class labels from the sel_IDs DataFrame
    c0 = list(sel_IDs.keys())[0]
    c1 = list(sel_IDs.keys())[1]
    
    # Patient/sample IDs by class
    rec_num_c0 = sel_IDs.iloc[:, 0]
    rec_num_c1 = sel_IDs.iloc[:, 1]
    
    # Merge and deduplicate all patient IDs
    all_patients = list(pd.concat([rec_num_c0, rec_num_c1]).sort_values().drop_duplicates().dropna().astype(str))
    
    # Initialize result containers
    dct_MODEL = {}
    dct_win_pred = {}
    dct_win_proba = {}
    dct_win_Xtrn = {}
    dct_win_Xtst = {}
    dct_win_ytrn = {}
    dct_win_ytst = {}
    models = {}
    feat_num = {}
    df_win_stype = pd.DataFrame(columns=['trn_len_c1', 'trn_len_c0', 'tst_len_c1', 'tst_len_c0'])
    
    for rs in range(iter_num):
        # try:
        # Group-aware train-test split
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        train_idx, test_idx = next(gss.split(all_patients, groups=all_patients))
        rec_num_train = [all_patients[i] for i in train_idx]
        rec_num_test = [all_patients[i] for i in test_idx]
        
        # Feature and label extraction
        X_train, y_train = get_features(rec_num_train, all_features, types)
        X_test, y_test = get_features(rec_num_test, all_features, types)
        
        # Standardization (if enabled)
        if standardize:
            scaler = StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        # Store intermediate results
        dct_win_Xtrn[rs] = X_train
        dct_win_Xtst[rs] = X_test
        dct_win_ytrn[rs] = y_train
        dct_win_ytst[rs] = y_test
        
        # Count samples per class
        df_win_stype.loc[rs, 'trn_len_c1'] = np.count_nonzero(y_train == c1)
        df_win_stype.loc[rs, 'trn_len_c0'] = np.count_nonzero(y_train == c0)
        df_win_stype.loc[rs, 'tst_len_c1'] = np.count_nonzero(y_test == c1)
        df_win_stype.loc[rs, 'tst_len_c0'] = np.count_nonzero(y_test == c0)
        
        print(f"model {rs}")
        # XGBoost requires binary labels (c0 -> 0, c1 -> 1)
        if model_name == "xgb":
            y_train_coded = np.where(y_train == c1, 1, 0).astype(int)
        else:
            y_train_coded = y_train
        
        # Chose and train initial model
        if model_name == "logreg":
            model = LogisticRegression(solver='liblinear')
            model.fit(X_train, y_train_coded)
        elif model_name == "rf":
            model = RandomForestClassifier(n_estimators=200, random_state=rs, )
            model.fit(X_train, y_train_coded)
        elif model_name == "xgb":
            model = XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='logloss',
                n_jobs=-1,
                random_state=rs,
            )
            model.fit(X_train, y_train_coded)
        
        # Perform Recursive Feature Elimination (if enough features)
        if X_train.shape[1] >= top_feat:
            rfe = RFE(
                estimator=model,
                n_features_to_select=top_feat,
                step=1
            )
            
            if model_name == "xgb":
                rfe.fit(X_train, y_train_coded)
                proba_pos = rfe.predict_proba(X_test)[:, 1]
                y_pred_bin = (proba_pos >= 0.5).astype(int)
                y_pred = np.where(y_pred_bin == 1, c1, c0)
                
                dct_win_pred[rs] = y_pred
                dct_win_proba[rs] = proba_pos
            
            else:
                rfe.fit(X_train, y_train)
                dct_win_pred[rs] = rfe.predict(X_test)
                dct_win_proba[rs] = rfe.predict_proba(X_test)[:, 1]
            
            models[rs] = rfe
            
            # Count selected features
            selected_features = X_train.columns[rfe.support_]
            for feat in selected_features:
                feat_num[feat] = feat_num.get(feat, 0) + 1
    
    # except Exception as e:
    #     print(f"[Warning] Iteration {rs} failed: {e}")
    #     continue
    
    # Convert feature frequency dict to DataFrame
    if feat_num:
        feat_num = pd.DataFrame([feat_num]).T
        feat_num = feat_num.sort_values(by=0, ascending=False)
    
    # Pack results
    dct_MODEL['dct_win_pred'] = [dct_win_pred]
    dct_MODEL['dct_win_proba'] = [dct_win_proba]
    dct_MODEL['dct_win_Xtrn'] = [dct_win_Xtrn]
    dct_MODEL['dct_win_Xtst'] = [dct_win_Xtst]
    dct_MODEL['dct_win_ytrn'] = [dct_win_ytrn]
    dct_MODEL['dct_win_ytst'] = [dct_win_ytst]
    dct_MODEL['df_win_stype'] = [df_win_stype]
    dct_MODEL['models'] = [models]
    
    return dct_MODEL, feat_num


if __name__ == '__main__':
    # === CONFIGURATION ===
    ITERATIONS = 1  # Number of iterations for modeling
    TOP_FEATURES = 10  # Number of top features to select
    TEST_SIZE = 0.2  # Proportion of data to use for testing
    WINDOW_INDEX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Leave empty for all window
    SIGNAL = ''  # Type of signal
    MODEL_NAME = "logreg"  # "logreg", "rf" or "xgb"
    DIR_NAME = "all_win0-9_top20"  # Name of directory where results are saved
    
    SELECTED_TYPES = [['False alarm', 'True alarm']]  # Types of samples to include
    STANDARDIZE = 0  # Whether to standardize features (1 = yes, 0 = no)
    COLOR_PALETTE = ['orange', 'purple']
    PRINT_RESULTS = 0
    PLOT_TITLE = 'False vs. True alarm'
    
    # === LOAD DATA ===
    # Read features from CSV
    all_features = pd.read_csv("biomarkers/all_bms_mean_windowed.csv").sort_values('Patient_id')
    # Select samples that matches the signal type (i.e. PPG/ECG)
    if SIGNAL != '':
        X = all_features.filter(regex=f"^({SIGNAL}|Patient_id$|Window_idx$)")
    else:
        X = all_features.drop(columns='Window_start_s')
    # Select window indices
    if WINDOW_INDEX != []:
        X = X[X["Window_idx"].isin(WINDOW_INDEX)]
    
    # Create Sample_id based on Patient_id and Window_idx
    widx = pd.to_numeric(X['Window_idx'], errors='raise').astype('Int64').astype(str)
    pid = X['Patient_id'].astype(str).str.strip()
    X.insert(0, 'Sample_id', pid + "_" + widx)
    # Delete Window_idx column
    X = X.drop(columns='Window_idx')
    
    # Locate NaN rows and delete them
    nan_rows = X.drop(columns=["Patient_id", "Sample_id"], errors="ignore").isna().all(axis=1)
    bad_ids = set(X.loc[nan_rows, "Sample_id"])
    X = X.loc[~nan_rows].reset_index(drop=True)
    
    # Replace infinity, -infinity and NaN with 0
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Read labels from CSV
    classes = pd.read_csv("biomarkers/ALARMS.txt")
    # Replace the numeric classes to string for consistency
    classes["Class"] = classes["Class"].map({0: "False alarm", 1: "True alarm"})
    # Load each class into lists
    ta_ids = classes.loc[classes["Class"].eq("True alarm"), "Patient_id"].tolist()
    fa_ids = classes.loc[classes["Class"].eq("False alarm"), "Patient_id"].tolist()
    ta_ids = [pid for pid in ta_ids if pid not in bad_ids]
    fa_ids = [pid for pid in fa_ids if pid not in bad_ids]
    # Connect classes into a dataframe where the column names are the classes and the column contains id-s for the class
    selected_ids = pd.DataFrame(
        zip_longest(fa_ids, ta_ids, fillvalue=pd.NA),
        columns=["False alarm", "True alarm"]
    )
    
    # === RUN MODELING ===
    
    # get_model is a custom function you have defined elsewhere.
    # It likely trains a classifier with cross-validation or feature selection.
    model_results, num_features = get_model(
        sel_IDs=selected_ids,
        test_size=TEST_SIZE,
        all_features=X,
        types=classes,
        standardize=STANDARDIZE,
        iter_num=ITERATIONS,
        top_feat=TOP_FEATURES,
        sel_type=SELECTED_TYPES,
        model_name=MODEL_NAME
    )
    
    # === ANALYZE WINDOW-LEVEL PREDICTIONS ===
    
    # Extract class labels (likely subject or group identifiers)
    classes = selected_ids.keys()
    
    # Displayed window title info (for ROC or other plots)
    window_title = f"{PLOT_TITLE}"
    
    # Create saving directory
    output_path = os.path.join('results', MODEL_NAME, DIR_NAME)
    os.makedirs(output_path, exist_ok=True)
    
    # Extract prediction results and compute ROC or other evaluation metrics
    prediction_results, roc_results = get_pred_res(
        d_ytst=model_results['dct_win_ytst'][0],
        d_pred=model_results['dct_win_pred'][0],
        d_proba=model_results['dct_win_proba'][0],
        title=window_title,
        prt=PRINT_RESULTS,
        color=COLOR_PALETTE[0],
        dname=os.path.join(MODEL_NAME, DIR_NAME),
        classes=classes,
        key=SIGNAL
    )
    
    # Save results
    prediction_results.to_csv(os.path.join(output_path, 'predictions.csv'), index=True)
    roc_results.to_csv(os.path.join(output_path, 'roc.csv'), index=True)
    num_features.to_csv(os.path.join(output_path, 'top_features.csv'), index=True)
    config = {'ITERATIONS': ITERATIONS, 'TOP_FEATURES': TOP_FEATURES, 'TEST_SIZE': TEST_SIZE,
              'WINDOWS': WINDOW_INDEX, 'MODEL_NAME': MODEL_NAME, 'SIGNAL': SIGNAL}
    configurations = pd.DataFrame(config)
    configurations.to_csv(os.path.join(output_path, 'configurations.csv'), index=False)