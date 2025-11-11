import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, matthews_corrcoef,  precision_recall_curve, roc_curve, auc, precision_score, recall_score
import matplotlib.pyplot as plt
from ast import literal_eval
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy.stats import bootstrap
from sklearn.utils import resample

def get_temp_df(res, split:str, c=2):
    """
    parameter c can be optimised.

    """
    temp = res.copy()

    p_e = []
    for p,e in list(zip(literal_eval(str(temp[f"{split}_pids"].values[0])),literal_eval(str(temp[f"{split}_eids"].values[0])))):
        pe = str(p)+"_"+str(e)
        #print(pe)
        p_e.append(pe)

    votedf = pd.DataFrame({"ID":p_e, 
                           "Label":literal_eval(str(temp[f"{split}_targets"].values[0])),
                           "pred_score": literal_eval(str(temp[f"{split}_probs"].values[0]))}
                           )
    
    
    p_mean = votedf.groupby(["ID"],as_index=False).mean()["pred_score"].values
    p_max = votedf.groupby(["ID"],as_index=False).max()["pred_score"].values
    ids = votedf.groupby(["ID"],as_index=False).max()["ID"].values
    n = votedf.groupby(["ID"],as_index=False).count()["pred_score"].values
    target = votedf.groupby(["ID"],as_index=False).max()["Label"].values
    
    temp = pd.DataFrame({"ID":ids,"target":target,"p_mean":p_mean,"p_max":p_max,"n":n})
    
    temp["p"] = temp.apply(lambda row: (row["p_max"]+row["p_mean"]*(row["n"]/c))/(1+(row["n"]/c)), axis=1)
    
    return temp

def get_temp_df_with_probs(df,pos_probs, labels, split):
    p_e = []
    ps = df[df.set==split].PatientDurableKey.values
    es = df[df.set==split].EncounterKey.values
    
    for p,e in list(zip(ps,es)):
        pe = str(p)+"_"+str(e)
        p_e.append(pe)
    
    votedf = pd.DataFrame({"ID":p_e, 
                           "Label":labels,
                           "pred_score": pos_probs})
    
    p_mean = votedf.groupby(["ID"],as_index=False).mean()["pred_score"].values
    p_max = votedf.groupby(["ID"],as_index=False).max()["pred_score"].values
    ids = votedf.groupby(["ID"],as_index=False).max()["ID"].values
    n = votedf.groupby(["ID"],as_index=False).count()["pred_score"].values
    target = votedf.groupby(["ID"],as_index=False).max()["Label"].values
    
    temp = pd.DataFrame({"ID":ids,"target":target,"p_mean":p_mean,"p_max":p_max,"n":n})
    
    c=2
    
    temp["p"] = temp.apply(lambda row: (row["p_max"]+row["p_mean"]*(row["n"]/c))/(1+(row["n"]/c)), axis=1)
    
    return temp

def plot_auc_curve(temp, modelname):
    fpr, tpr, thresholds = roc_curve(temp.target, temp.p_mean)
    auc_score = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Val (area = {:.3f})'.format(auc_score))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'ROC curve {modelname}')
    plt.legend(loc='best')
    plt.show()


def compute_mcc(true_labels, pred_labels):
    """
    From: https://stackoverflow.com/questions/52370048/calculating-matthew-correlation-coefficient-for-a-matrix-takes-too-long
    
    Compute matthew's correlation coefficient.
    :param true_labels: 2D integer array (features x samples)
    :param pred_labels: 2D integer array (features x samples)
    :return: mcc (samples1 x samples2)
    """
    # prep inputs for confusion matrix calculations
    pred_labels_1 = pred_labels == 1; pred_labels_0 = pred_labels == 0
    true_labels_1 = true_labels == 1; true_labels_0 = true_labels == 0
    
    # dot product of binary matrices
    confusion_dot = lambda a,b: np.dot(a.T.astype(int), b.astype(int)).T
    TP = confusion_dot(pred_labels_1, true_labels_1)
    TN = confusion_dot(pred_labels_0, true_labels_0)
    FP = confusion_dot(pred_labels_1, true_labels_0)
    FN = confusion_dot(pred_labels_0, true_labels_1)

    mcc = np.array((TP * TN) - (FP * FN),dtype='float')
    denom = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return np.divide(mcc, denom, out=np.zeros_like(mcc), where=denom!=0)

def mccf1_metric(mcc,dist, W_num_of_subranges = 100):
    # https://github.com/krishnanlab/MCC-F1-Curve-and-Metrics/blob/master/MCC_F1_funcs.py
    subrange_intervals = np.linspace(np.min(mcc), np.max(mcc),W_num_of_subranges) # breaking into subranges
    
    # Computing MCC values per subrange
    values_per_subrange = np.zeros_like(subrange_intervals) # 'n' is the number of points per subrange
    for i in range(W_num_of_subranges-1):
        for j in mcc:
            if j >= subrange_intervals[i] and j < subrange_intervals[i+1]:
                values_per_subrange[i] = values_per_subrange[i] + 1
    
    Distance_of_points_within_subrange = dist
    sum_of_distance_within_subrange = np.zeros_like(subrange_intervals)
    index = -1
    for value in mcc:
        index += 1
        for i in range(W_num_of_subranges-1):
                 if value >= subrange_intervals[i] and value < subrange_intervals[i+1]:
                        sum_of_distance_within_subrange[i] = sum_of_distance_within_subrange[i] + Distance_of_points_within_subrange[index]

    # Mean Distance across subranges
    mean_Distance_per_subrange = np.array(sum_of_distance_within_subrange/values_per_subrange)
    total_number_of_subranges = 0  
    total_sum_of_mean_distances = 0
    for i in mean_Distance_per_subrange:
        if not np.isnan(i):
            total_number_of_subranges += 1 
            total_sum_of_mean_distances += i # addition of all the means across subranges that have atleast 1 MCC value.

    grand_mean_distance = total_sum_of_mean_distances/total_number_of_subranges # P = total number of subranges that have atleast 1 MCC value
   
    """ 
    Compare the grand average distance to √2 (The distance between the point of worst performance (0,0) and 
    the point of perfect performance (1,1) is √2).That is the maximum possible distance between a point on the MCC-𝐹1 curve
    The ratio between the grand avgerage distance and √2 is taken.
    This ratio ranges between 0 and 1 (worst value = 0; best value = 1). To get the MCC-𝐹1 score, we subtract this ratio from 1
    """
    MCC_F1_Met = 1 - (grand_mean_distance /np.sqrt(2))
    return MCC_F1_Met


def mccf1_threshold(y_true, y_prob, plot_curve=True, num_thr_plot=5):
    thresholds = np.sort(np.unique(y_prob))[::-1]
    mcc = []
    f1 = []
    # Unit-normalized MCC and F1-score for each threshold
    for thr in tqdm(thresholds):
        y_pred = y_prob >= thr
        mcc.append((compute_mcc(y_true, y_pred)+1)*0.5)
        f1.append(f1_score(y_true, y_pred))
    mcc = np.array(mcc)
    f1 = np.array(f1)
    
    # Distance to perfect point
    dist = np.sqrt((mcc-1)**2 + (f1-1)**2)
    best_thr = np.argmin(dist)
    mccf1 = mccf1_metric(mcc,dist)
    # Plot MCC-F1 curve
    if plot_curve:
        idx_jump = int(len(f1)/num_thr_plot)
        fig, ax = plt.subplots(figsize = (10,8))
        ax.set_clip_on(False)
        ax.scatter(f1, mcc, alpha=0.5)
        
        for i, thr in enumerate(thresholds):
            if i % idx_jump== 0:
                ax.plot(f1[i], mcc[i], 'o', color='red')
                ax.annotate(r'$\tau = $' + str(round(thr, 5)), (f1[i], mcc[i]+.05*mcc[i]), fontsize = 10, color = 'red')
        
        ax.plot(f1[best_thr], mcc[best_thr], 'o', color = 'lime')
        ax.annotate(r'$\tau = $' + str(round(thresholds[best_thr], 5)), (f1[best_thr] + .02*f1[best_thr], mcc[best_thr]-.01*mcc[best_thr]), fontsize = 14, color = 'green')
        
        ax.plot(1,1, 'o', color = 'black')
        ax.annotate('point of perfect performance', (0.7, 0.95), fontsize = 12)
        
        ax.plot(0,0, 'o', color = 'black')
        ax.annotate('point worst performance', (0, 0.05), fontsize = 12)
        
        P = y_true.sum()
        N = len(y_true) - P
        ground_truth = 2*(P / (2*P + N))
        ax.plot([0, ground_truth], [0.5, 0.5], color = 'black', linestyle = '--')
        ax.annotate('random line', (ground_truth / 4, 0.45))
        
        ax.set(title=f'MCC-F1 curve', xlabel='F1-score', ylabel='unit-normalized MCC')
                
    return thresholds[np.argmin(dist)],mccf1


def compute_specificity(targets, preds):
    return recall_score(targets, preds, pos_label=0)

def compute_auroc(targets,preds):
    fpr, tpr, thresholds = roc_curve(targets,preds)
    auc_score = auc(fpr, tpr)
    return auc_score

def compute_auprc(targets,preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    area = auc(recall,precision)
    return area

def compute_f1_neg(targets,preds):
    return f1_score(targets,preds,pos_label=0)

def compute_f1_weighted(targets,preds):
    return f1_score(targets,preds,average="weighted")


def stratified_bootstrap_mcc(targs, preds, seed: int = 22, n_bootstraps: int = 1000, ci: float = 0.05, alpha: float = 0.025, beta=1.0, only_scores = False):
    # https://github.com/SocialComplexityLab/life2vec/blob/5601191a08a3feae97b4641776692916400c8ed7/analysis/metric/survival.ipynb#L667
    preds = np.array(preds)
    targs = np.array(targs)

    ids = np.arange(0, targs.shape[0], 1)
    
    idx = list()
    for n in range(n_bootstraps):
        i  = resample(ids, stratify=targs, random_state=n)
        if len(np.unique(targs[i])) < 2:
                continue
        idx.append(i)
    
    scores = np.array([compute_mcc(targs[i], preds[i]) for i in idx])
    
    if only_scores:
        return scores
    
    return {"mean": compute_mcc(targs, preds), "lower": np.quantile(scores, ci /2), "upper": np.quantile(scores, 1-ci/2)}

def compute_metrics(temp, t, bootstrapping=True):

    auroc = np.round(compute_auroc(temp.target,temp.p_mean),3)
    if bootstrapping:
        conf = bootstrap((temp.target.values,temp.p_mean.values), 
                        compute_auroc, vectorized=False, paired=True,
                        random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"AUROC = {auroc} [{low_conf},{high_conf}]")

    auprc = np.round(compute_auprc(temp.target,temp.p_mean),3)
    if bootstrapping:
        conf = bootstrap((temp.target,temp.p_mean), 
                        compute_auprc, vectorized=False, paired=True,
                        random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"AUPRC = {auprc} [{low_conf},{high_conf}]")
    
    print("mcc-f1 threshold=",np.round(t,3))
    preds = [1 if i>=t else 0 for i in temp.p_mean]
    targets = temp.target

    mcc = np.round(matthews_corrcoef(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), compute_mcc, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"MCC = {mcc} [{low_conf},{high_conf}]")
        print("stratified bootstrapping mcc:",stratified_bootstrap_mcc(targets,preds))

    f1_pos = np.round(f1_score(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), f1_score, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"F1 Pos = {f1_pos} [{low_conf},{high_conf}]")
    
    f1_neg = np.round(compute_f1_neg(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), compute_f1_neg, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"F1 Neg = {f1_neg} [{low_conf},{high_conf}]")
    
    f1_weighted = np.round(compute_f1_weighted(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), compute_f1_weighted, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"F1 weighted avg = {f1_weighted} [{low_conf},{high_conf}]")

    precision = np.round(precision_score(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), precision_score, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"precision = {precision} [{low_conf},{high_conf}]")

    recall = np.round(recall_score(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), recall_score, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"recall = {recall} [{low_conf},{high_conf}]")

    specificity = np.round(compute_specificity(targets,preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), compute_specificity, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"specificity = {specificity} [{low_conf},{high_conf}]")

    accuracy = np.round(accuracy_score(targets,preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), accuracy_score, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"accuracy = {accuracy} [{low_conf},{high_conf}]")
    
    results = {"mcc-f1 threshold":t,
               "auroc":auroc,
               "auprc":auprc, 
               "mcc":mcc, 
               "f1":f1_pos,
               "f1 neg": f1_neg,
               "f1 weighted avg": f1_weighted,
               "precision":precision, 
               "recall":recall,
               "specificity": specificity,
               "accuracy": accuracy}
    print(results)

    M = confusion_matrix(targets,preds)
    tn, fp, fn, tp = M.ravel()

    fig, ax1 = plt.subplots(1,1, figsize = (5,4))
    
    sns.heatmap(M, annot=True, fmt='d',vmin=0, vmax=sum(M.ravel()), ax=ax1)
    ax1.set_xlabel("Predicted label", fontsize=14, labelpad=20);
    ax1.set_ylabel("True label", fontsize=14, labelpad=20);
    plt.show()


def compute_metrics_v2(target, pred, t, bootstrapping=True):

    auroc = np.round(compute_auroc(target,pred),3)
    if bootstrapping:
        conf = bootstrap((target.values,pred.values), 
                        compute_auroc, vectorized=False, paired=True,
                        random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"AUROC = {auroc} [{low_conf},{high_conf}]")

    auprc = np.round(compute_auprc(target,pred),3)
    if bootstrapping:
        conf = bootstrap((target,pred), 
                        compute_auprc, vectorized=False, paired=True,
                        random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"AUPRC = {auprc} [{low_conf},{high_conf}]")
    
    print("mcc-f1 threshold=",np.round(t,3))
    preds = [1 if i>=t else 0 for i in pred]
    #preds_05 = [1 if i>=0.5 else 0 for i in pred]
    targets = target

    mcc = np.round(matthews_corrcoef(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), compute_mcc, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"MCC = {mcc} [{low_conf},{high_conf}]")
        print("stratified bootstrapping mcc:",stratified_bootstrap_mcc(targets,preds))

    f1_pos = np.round(f1_score(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), f1_score, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"F1 Pos = {f1_pos} [{low_conf},{high_conf}]")
    
    f1_neg = np.round(compute_f1_neg(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), compute_f1_neg, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"F1 Neg = {f1_neg} [{low_conf},{high_conf}]")
    
    f1_weighted = np.round(compute_f1_weighted(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), compute_f1_weighted, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"F1 weighted avg = {f1_weighted} [{low_conf},{high_conf}]")

    precision = np.round(precision_score(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), precision_score, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"precision = {precision} [{low_conf},{high_conf}]")

    recall = np.round(recall_score(targets, preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), recall_score, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"recall = {recall} [{low_conf},{high_conf}]")

    specificity = np.round(compute_specificity(targets,preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), compute_specificity, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"specificity = {specificity} [{low_conf},{high_conf}]")

    accuracy = np.round(accuracy_score(targets,preds),3)
    if bootstrapping:
        conf = bootstrap((targets, preds), accuracy_score, vectorized=False, paired=True,
                    random_state=22, n_resamples=1000)
        low_conf, high_conf = np.round(conf.confidence_interval[0],3), np.round(conf.confidence_interval[1],3)
        print(f"accuracy = {accuracy} [{low_conf},{high_conf}]")
    
    results = {"mcc-f1 threshold":t,
               "auroc":auroc,
               "auprc":auprc, 
               "mcc":mcc, 
               "f1":f1_pos,
               "f1 neg": f1_neg,
               "f1 weighted avg": f1_weighted,
               "precision":precision, 
               "recall":recall,
               "specificity": specificity,
               "accuracy": accuracy}
    print(results)

def MCCF1_threshold_and_metrics(temp_train, temp, threshold=None, show_train_performance=True):
    
    """
    MCC-F1 thresholds calculated on trainset. The rest calculated on val or test.
    temp_train: pandas dataframe with softmax probabilities on TRAIN grouped on encounters. Here we use the mean score per encounter, other options are p_max and p, the latter a weighted average.
    temp: pandas dataframe with softmax probabilities on VAL or TEST grouped on encounters. 
    
    """
    fpr, tpr, thresholds = roc_curve(temp.target, temp.p_mean)
    auc_score = auc(fpr, tpr)
    print(f'AUROC score: {auc_score}')

    if threshold==None:
        t,mccf1_score = mccf1_threshold(temp_train.target, temp_train.p_mean)
        print(f'MCC-F1 threshold from TRAIN t={t}')
        print(f'MCC-F1 score TRAIN= {mccf1_score}')
    else:
        t=threshold
    
    print()
    compute_metrics(temp,t)
    print()

    if show_train_performance:
        print("\n TRAIN PERFORMANCE")
        compute_metrics(temp_train,t,bootstrapping=False)
