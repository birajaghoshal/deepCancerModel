import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
from PIL import Image

def get_error(pred, true):
    n_bootstraps = 1000
    rng_seed0 = 42  # control reproducibility
    rng_seed1 = 100 # control reproducibility
    rng_seed2 = 250  # control reproducibility
    rng_seed3 = 400  # control reproducibility

    bootstrapped_scores0 = []
    bootstrapped_scores1 = []
    bootstrapped_scores2 = []
    bootstrapped_scores3 = []

    rng0 = np.random.RandomState(rng_seed0)
    rng1 = np.random.RandomState(rng_seed1)
    rng2 = np.random.RandomState(rng_seed2)
    rng3 = np.random.RandomState(rng_seed3)

    true_all=true.ravel()
    pred_all=pred.ravel()


    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices0 = rng0.random_integers(0, len(pred[:,0]) - 1, len(pred[:,0]))
        if len(np.unique(true[indices0,0])) < 2:
            continue
        indices1 = rng1.random_integers(0, len(pred[:,1]) - 1, len(pred[:,1]))
        if len(np.unique(true[indices1,1])) < 2:
            continue
        indices2 = rng2.random_integers(0, len(pred[:,2]) - 1, len(pred[:,2]))
        if len(np.unique(true[indices2,2])) < 2:
            continue
        indices3 = rng3.random_integers(0, len(pred_all) - 1, len(pred_all))
        if len(np.unique(true_all[indices3])) < 2:
            continue
       

        score0 = roc_auc_score(true[indices0,0], pred[indices0,0])
        score1 = roc_auc_score(true[indices1,1], pred[indices1,1])
        score2 = roc_auc_score(true[indices2,2], pred[indices2,2])
        score3 = roc_auc_score(true_all[indices3], pred_all[indices3])

        bootstrapped_scores0.append(score0)
        bootstrapped_scores1.append(score1)
        bootstrapped_scores2.append(score2)
        bootstrapped_scores3.append(score3)

    sorted_scores0 = np.array(bootstrapped_scores0)
    sorted_scores0.sort()
    confidence_lower0 = sorted_scores0[int(0.05 * len(sorted_scores0))]
    confidence_upper0 = sorted_scores0[int(0.95 * len(sorted_scores0))]

    sorted_scores1 = np.array(bootstrapped_scores1)
    sorted_scores1.sort()
    confidence_lower1 = sorted_scores1[int(0.05 * len(sorted_scores1))]
    confidence_upper1 = sorted_scores1[int(0.95 * len(sorted_scores1))]

    sorted_scores2 = np.array(bootstrapped_scores2)
    sorted_scores2.sort()
    confidence_lower2 = sorted_scores2[int(0.05 * len(sorted_scores2))]
    confidence_upper2 = sorted_scores2[int(0.95 * len(sorted_scores2))]

    #micro
    sorted_scores3=np.array(bootstrapped_scores3)                                      
    sorted_scores3.sort()
                                      
    confidence_lower_micro = sorted_scores3[int(0.05 * len(sorted_scores3))]
    confidence_upper_micro = sorted_scores3[int(0.95 * len(sorted_scores3))]                                    


    #macro
                                       
    sorted_scores4=np.array(np.concatenate((bootstrapped_scores0,bootstrapped_scores1,bootstrapped_scores2), axis=0))                                      
    sorted_scores4.sort()
                                      
    confidence_lower_macro = sorted_scores4[int(0.05 * len(sorted_scores4))]
    confidence_upper_macro = sorted_scores4[int(0.95 * len(sorted_scores4))]                                    

    return confidence_lower0, confidence_upper0, confidence_lower1, confidence_upper1,confidence_lower2, confidence_upper2,confidence_lower_macro, confidence_upper_macro,confidence_lower_micro, confidence_upper_micro



def get_auc(path, predictions, labels, classes=[0, 1, 2]):

    """
    Given predictions and labels, return the AUCs for all classes
    and micro, macro AUCs. Also saves a plot of the ROC curve to the
    path.

    """

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    if len(classes) > 2:
        # Convert labels to one-hot-encoding
        labels = label_binarize(labels, classes = classes)

        ### Individual class AUC ###
        for i in classes:
            fpr[i], tpr[i], _ = roc_curve(labels[:, i], predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        errors=get_error(predictions,labels)
	
        roc_auc["cu0"]=errors[1]
        roc_auc["cu1"]=errors[3]
        roc_auc["cu2"]=errors[5]
   
        roc_auc["cl0"]=errors[0]
        roc_auc["cl1"]=errors[2]
        roc_auc["cl2"]=errors[4]
   
        ### Micro AUC ###
        fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        roc_auc["micro_cl"]=errors[8]
        roc_auc["micro_cu"]=errors[9]


        ### Macro AUC ###
        all_fpr = np.unique(np.concatenate([fpr[i] for i in classes]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in classes:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(classes)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"]) 
        roc_auc["macro_cl"]=errors[6]      
        roc_auc["macro_cu"]=errors[7]

        ### Make plot ###
        plt.figure(figsize=(12, 8))
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(classes, colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
    else:
        fpr, tpr, _ = roc_curve(labels, predictions[:,1])
        auc_result = auc(fpr, tpr)

        for i in list(classes) + ['macro', 'micro']:
            roc_auc[i] = auc_result

        plt.figure(figsize=(12, 8))
        plt.plot(fpr, tpr, lw=2,
                 label='ROC curve (area = {0:0.2f})'
                 ''.format(auc_result))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(path)

    return roc_auc
