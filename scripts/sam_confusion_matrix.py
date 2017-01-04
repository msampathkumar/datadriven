"""Welcome to sam_confusion_matrix.py.

This is module is to create a maxtrix plot
 of the columns which are having categorical values.
"""

from __future__ import division

import itertools

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def sam_plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
    """To prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    plots_dims = itertools.product(list(range(cm.shape[0])),
                                   list(range(cm.shape[1])))
    for i, j in plots_dims:
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def sam_confusion_maxtrix(y_test, y_pred, class_names):
    """Print confusion matrix for inputdata & classes.

    Args:
        * y_test(array): list of actual values
        * y_pred(array): list of predicted values
        * class_name(array): list of class names

    Example
    >>> sam_confusion_maxtrix(y_test,
                                y_pred,
                                class_names=RAW_y.status_group.value_counts().keys()
                                ):
    """
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure(figsize=(8, 8))
    sam_plot_confusion_matrix(cnf_matrix,
                              classes=class_names,
                              title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 8))
    sam_plot_confusion_matrix(cnf_matrix,
                              classes=class_names,
                              normalize=True,
                              title='Normalized confusion matrix')
    plt.show()
