from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score,average_precision_score,roc_auc_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt

def eval_sklearn_model(model,feature_matrix,target_col,random_state=0):

    X_train, X_test, y_train, y_test = train_test_split(feature_matrix,
                                                        target_col,
                                                        stratify=target_col, 
                                                        test_size=0.2,
                                                        random_state=random_state)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    
    model_f1 = f1_score(y_test, y_pred)
    model_prec = precision_score(y_test, y_pred)
    model_rec = recall_score(y_test, y_pred)
    model_auprc = average_precision_score(y_test,y_pred_prob)
    model_auroc = roc_auc_score(y_test,y_pred_prob)
    
    model_f1_micro = f1_score(y_test, y_pred, average='micro')
    model_prec_micro = precision_score(y_test, y_pred, average='micro')
    model_rec_micro = recall_score(y_test, y_pred, average='micro')
    model_auprc_micro = average_precision_score(y_test,y_pred_prob,average='micro')
    model_auroc_micro = roc_auc_score(y_test,y_pred_prob,average='micro')
    
    model_f1_macro = f1_score(y_test, y_pred, average='macro')
    model_prec_macro = precision_score(y_test, y_pred, average='macro')
    model_rec_macro = recall_score(y_test, y_pred, average='macro')
    model_auprc_macro = average_precision_score(y_test,y_pred_prob,average='macro')
    model_auroc_macro = roc_auc_score(y_test,y_pred_prob,average='macro')
    
    report = {
        "binary": {
            "f1": model_f1,
            "precision": model_prec,
            "recall": model_rec,
            "AUPRC": model_auprc,
            "AUROC": model_auroc
        },
        "micro": {
            "f1": model_f1_micro,
            "precision": model_prec_micro,
            "recall": model_rec_micro,
            "AUPRC": model_auprc_micro,
            "AUROC": model_auroc_micro
        },
        "macro": {
            "f1": model_f1_macro,
            "precision": model_prec_macro,
            "recall": model_rec_macro,
            "AUPRC": model_auprc_macro,
            "AUROC": model_auroc_macro
        }
    }
    
    return report

# Code partially modified from:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

# def draw_roc(feature_matrix,target_col,n_split=5,max_iter=MAX_ITER):
#     cv = StratifiedKFold(n_splits=n_split,random_state=RANDOM_SEED)
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)

#     lr_clf = LogisticRegression(random_state=RANDOM_SEED,max_iter=max_iter)

#     fig, ax = plt.subplots()

#     for i, (train, test) in enumerate(cv.split(feature_matrix, target_col)):
#         print(f"LR Iteration: {i}")
#         lr_clf.fit(feature_matrix[train], target_col[train])
#         viz = plot_roc_curve(lr_clf, feature_matrix[test], target_col[test],
#                              name='ROC fold {}'.format(i),
#                              alpha=0.3, lw=1, ax=ax)
#         interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         aucs.append(viz.roc_auc)

#     ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#             label='Chance', alpha=.8)

#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     ax.plot(mean_fpr, mean_tpr, color='b',
#             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#             lw=2, alpha=.8)

#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')

#     ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
#            title="Receiver operating characteristic example")
#     ax.legend(loc="lower right")
#     plt.show()