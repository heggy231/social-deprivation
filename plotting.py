# -*- coding: utf-8 -*-
"""
Collected functions useful for plotting ROC, scree, etc.
"""

from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import (
    cluster,
    datasets,
    decomposition,
    ensemble,
    lda,
    manifold,
    random_projection,
    preprocessing)
import numpy as np
from sklearn.metrics import roc_curve, f1_score, classification_report, accuracy_score, confusion_matrix


def scree_plot(num_components, pca):
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35,
           color=[(0.949, 0.718, 0.004),
                  (0.898, 0.49, 0.016),
                  (0.863, 0, 0.188),
                  (0.694, 0, 0.345),
                  (0.486, 0.216, 0.541),
                  (0.204, 0.396, 0.667),
                  (0.035, 0.635, 0.459),
                  (0.486, 0.722, 0.329),
                  ])

    ax.annotate(
        r"%d%%" %
        (int(
            vals[0] *
            100)),
        (ind[0] +
         0.2,
         vals[0]),
        va="bottom",
        ha="center",
        fontsize=12)
    ax.annotate(
        r"%d%%" %
        (int(
            vals[1] *
            100)),
        (ind[1] +
         0.2,
         vals[1]),
        va="bottom",
        ha="center",
        fontsize=12)
    ax.annotate(
        r"%d%%" %
        (int(
            vals[2] *
            100)),
        (ind[2] +
         0.2,
         vals[2]),
        va="bottom",
        ha="center",
        fontsize=12)
    ax.annotate(
        r"%d%%" %
        (int(
            vals[3] *
            100)),
        (ind[3] +
         0.2,
         vals[3]),
        va="bottom",
        ha="center",
        fontsize=12)
    ax.annotate(
        r"%d%%" %
        (int(
            vals[4] *
            100)),
        (ind[4] +
         0.2,
         vals[4]),
        va="bottom",
        ha="center",
        fontsize=12)
    ax.annotate(
        r"%d%%" %
        (int(
            vals[5] *
            100)),
        (ind[5] +
         0.2,
         vals[5]),
        va="bottom",
        ha="center",
        fontsize=12)
    ax.annotate(r"%s%%" % ((str(vals[6] * 100)[:4 + (0 - 1)])),
                (ind[6] + 0.2, vals[6]), va="bottom", ha="center", fontsize=12)
    ax.annotate(r"%s%%" % ((str(vals[7] * 100)[:4 + (0 - 1)])),
                (ind[7] + 0.2, vals[7]), va="bottom", ha="center", fontsize=12)

    ax.set_xticklabels(ind,
                       fontsize=12)
    ax.set_yticklabels(
        ('0.00',
         '0.05',
         '0.10',
         '0.15',
         '0.20',
         '0.25'),
        fontsize=12)
    ax.set_ylim(0, .25)
    ax.set_xlim(0 - 0.45, 8 + 0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)
    plt.title("Scree Plot for the Digits Dataset", fontsize=16)


def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(1 * y[i]),
                 fontdict={'weight': 'bold', 'size': 12})

    plt.xticks([]), plt.yticks([])
    plt.ylim([-0.1, 1.1])
    plt.xlim([-0.1, 1.1])

    if title is not None:
        plt.title(title, fontsize=16)


def plot_roc(X, y, clf_class, n_folds=5, **kwargs):
    plt.figure(1, figsize=(12, 12))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = KFold(len(y), n_folds, shuffle=True)
    y_prob = np.zeros((len(y), 2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' %
            (i, roc_auc))
#        predictions = clf.predict(X_test)
#        print 'accuracy: ', accuracy_score(y_test,predictions)
#        print classification_report(y_test, predictions)
#        print 'confusion matrix: '
#        print confusion_matrix(y_test,predictions)
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(
        mean_fpr,
        mean_tpr,
        'k--',
        label='Mean ROC (area = %0.2f)' %
        mean_auc,
        lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
