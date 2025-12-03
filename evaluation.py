import matplotlib.pyplot as plt
import os
import os.path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc

def plot_ROC(fpr_median, tpr_median, auroc_median, tpr_q1, tpr_q3, color, title, save_dir, model_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve with shaded interquartile range (Q1–Q3).

    Parameters:
    ----------
    fpr_median : array-like
        Median False Positive Rate values (1 - specificity).

    tpr_median : array-like
        Median True Positive Rate values (sensitivity).

    auroc_median : float
        Area Under the ROC Curve (AUROC) for the median ROC.

    tpr_q1 : array-like
        25th percentile (Q1) of True Positive Rates at each FPR point.

    tpr_q3 : array-like
        75th percentile (Q3) of True Positive Rates at each FPR point.

    color : str
        Color to use for the interquartile shading.

    title : str
        Title of the plot and base name for the saved figure.

    save_dir : str
        Subdirectory within `save_fig/` to save the plot.

    model_label : str
        Label for the ROC curve (used in legend).

    Returns:
    -------
    None
        The function displays the plot and saves it as a PNG file.
    """

    # Configure plot style
    plt.rcParams.update({
        'font.size': 24,
        'axes.titlesize': 24,
        'axes.labelsize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 18
    })

    # Create figure
    plt.figure(figsize=(10, 10))

    # Plot the median ROC curve
    plt.plot(fpr_median, tpr_median, color='black', linewidth=2,
             label=f'{model_label} Median (AUROC = {auroc_median:.2f})')

    # Fill between Q1 and Q3 to show interquartile range
    plt.fill_between(fpr_median, tpr_q1, tpr_q3, color=color, alpha=0.5, label='Q1–Q3 Range')

    # Plot the diagonal (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    # Annotate plot
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)

    # Prepare save path
    filename = f'ROC_{title.replace(" ", "_")}.png'
    output_path = os.path.join('results', save_dir)
    os.makedirs(output_path, exist_ok=True)

    # Save and show the plot
    plt.savefig(os.path.join(output_path, filename), bbox_inches='tight')
    plt.show()
    
def get_pred_res(d_ytst, d_pred, d_proba, title, prt, color, dname, classes, key):
    """
    Function to compute and display various performance metrics and plots for model predictions.

    Parameters:
    - d_ytst: dict, actual target values for the test set (in dictionary form)
    - d_pred: dict, predicted target values (in dictionary form)
    - d_proba: dict, predicted probabilities (in dictionary form)
    - title: str, the title to be used for the plot
    - prt: bool, flag to determine if metrics should be printed
    - color: str, color to use for the plot
    - dname: str, directory name
    - classes: list, list of the class labels (e.g., [0, 1])
    - key: str, some key for identification in output

    Returns:
    - PRED_RES: DataFrame containing individual model results
    - ROC_RES: DataFrame containing aggregated results (mean, median)
    """

    PRED_RES = pd.DataFrame(columns=['AUROC', 'PREC', 'SEN', 'SPEC', 'F1'])

    all_fpr = {}
    all_tpr = {}

    for rs in d_ytst.keys():
        try:
            y_test = d_ytst[rs]
            y_pred = d_pred[rs]
            y_proba = d_proba[rs]

            # Convert class labels to binary (0 or 1)
            ind_tc0 = np.where(y_test == classes[0])
            ind_tc1 = np.where(y_test == classes[1])
            y_test[ind_tc0] = 0
            y_test[ind_tc1] = 1
            y_test = y_test.astype(int)

            ind_pc0 = np.where(y_pred == classes[0])
            ind_pc1 = np.where(y_pred == classes[1])
            y_pred[ind_pc0] = 0
            y_pred[ind_pc1] = 1
            y_pred = y_pred.astype(int)

            # Calculate confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            if len(conf_matrix) == 1:
                tmp_confm = np.zeros((2, 2), dtype=int)
                tmp_confm[0][0] = conf_matrix[0]
                conf_matrix = tmp_confm

            # Calculate precision, recall, and F1 score
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Calculate specificity (Sp)
            tn, fp, fn, tp = conf_matrix.ravel()

            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)

            # Calculate AUROC
            auroc = auc(fpr, tpr)

            # Calculate mean results
            all_fpr[rs] = fpr
            all_tpr[rs] = tpr

            # Add elements to results dataframe
            PRED_RES.loc[rs, 'AUROC'] = auroc
            PRED_RES.loc[rs, 'PREC'] = precision
            PRED_RES.loc[rs, 'SEN'] = sensitivity
            PRED_RES.loc[rs, 'SPEC'] = specificity
            PRED_RES.loc[rs, 'F1'] = f1

            # Print the results
            if prt:
                print('PPG Stroke win pred:')
                print(f'AUROC: {auroc:.2f}')
                print(f'Precision: {precision:.2f}')
                print(f'Sensitivity: {sensitivity:.2f}')
                print(f'Specificity: {specificity:.2f}')
                print(f'F1 Score: {f1:.2f}')
        except Exception as e:
            pass

    # Uniform FPR and TPR
    new_fprs = pd.DataFrame()
    new_tprs = pd.DataFrame()
    len_w = 50
    for tmpk in all_fpr.keys():
        tmp_fpr = all_fpr[tmpk]
        tmp_tpr = all_tpr[tmpk]
        new_fprs[tmpk] = np.interp(np.linspace(0, len(tmp_fpr) - 1, len_w), np.arange(len(tmp_fpr)), tmp_fpr)
        new_tprs[tmpk] = np.interp(np.linspace(0, len(tmp_tpr) - 1, len_w), np.arange(len(tmp_tpr)), tmp_tpr)

    FPR_med = new_fprs.T.median()
    TPR_med = new_tprs.T.median()
    AUC_med = auc(FPR_med, TPR_med)
    PRE_med = TPR_med / (TPR_med + FPR_med)
    F1_med = 2 * (PRE_med * TPR_med) / (PRE_med + TPR_med)

    all_auc = []
    for ti in range(new_fprs.shape[1]):
        tmp_fpr = new_fprs.iloc[:, ti]
        tmp_tpr = new_tprs.iloc[:, ti]
        tmp_auc = auc(tmp_fpr, tmp_tpr)
        all_auc.append(tmp_auc)

    AUC_Q1 = np.percentile(all_auc, 25)
    AUC_Q3 = np.percentile(all_auc, 75)
    AUC_std = np.std(all_auc)

    # Find the index of the threshold with maximum F1 score
    index_max_f1 = np.argmax(F1_med)
    Max_F1_med = F1_med[index_max_f1]
    Max_Pre_med = PRE_med[index_max_f1]

    TPR_Q1 = new_tprs.T.quantile(0.25)
    TPR_Q3 = new_tprs.T.quantile(0.75)

    # Plot ROC curve
    plot_ROC(FPR_med, TPR_med, AUC_med, TPR_Q1, TPR_Q3, color, title, dname, key)

    youden_j = TPR_med - FPR_med
    index_max_j = np.argmax(youden_j)
    SEN_med = TPR_med[index_max_j]
    SPE_med = 1 - FPR_med[index_max_j]

    FPR_mean = new_fprs.T.mean()
    TPR_mean = new_tprs.T.mean()
    AUC_mean = auc(FPR_mean, TPR_mean)
    PRE_mean = TPR_mean / (TPR_mean + FPR_mean)
    F1_mean = 2 * (PRE_mean * TPR_mean) / (PRE_mean + TPR_mean)

    # Find the index of the threshold with maximum F1 score
    index_max_f1 = np.argmax(F1_mean)
    Max_F1_mean = F1_mean[index_max_f1]
    Max_Pre_mean = PRE_mean[index_max_f1]

    youden_j = TPR_mean - FPR_mean
    index_max_j = np.argmax(youden_j)
    SEN_mean = TPR_mean[index_max_j]
    SPE_mean = 1 - FPR_mean[index_max_j]

    if prt:
        print('Median')
        print(f'AUC: {AUC_med:.2f} ({AUC_Q1:.2f}-{AUC_Q3:.2f})')
        print(f'PRE: {Max_Pre_med:.2f}')
        print(f'SEN: {SEN_med:.2f}')
        print(f'SPE: {SPE_med:.2f}')
        print(f'F1: {Max_F1_med:.2f}')
        print('Mean')
        print(f'AUC: {AUC_mean:.2f} \u00B1 {AUC_std:.2f}')
        print(f'PRE: {Max_Pre_mean:.2f}')
        print(f'SEN: {SEN_mean:.2f}')
        print(f'SPE: {SPE_mean:.2f}')
        print(f'F1: {Max_F1_mean:.2f}')

    # Add results to a new DataFrame for reporting
    ROC_RES = pd.DataFrame({
        'Mean': [
            f"{AUC_mean:.2f} ± {AUC_std:.2f}",
            f"{Max_Pre_mean:.2f}",
            f"{SEN_mean:.2f}",
            f"{SPE_mean:.2f}",
            f"{Max_F1_mean:.2f}"
        ],
        'Median': [
            f"{AUC_med:.2f} ({AUC_Q1:.2f}-{AUC_Q3:.2f})",
            f"{Max_Pre_med:.2f}",
            f"{SEN_med:.2f}",
            f"{SPE_med:.2f}",
            f"{Max_F1_med:.2f}"
        ]
    }, index=['AUC', 'Precision', 'Sensitivity', 'Specificity', 'F1'])

    return PRED_RES, ROC_RES