"""Script for all local tools."""
import pickle
import numpy as np
import pandas as pd
import sklearn.metrics as sk_metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from collections import defaultdict
from sklearn import preprocessing


def sam_pickle_save(df_x, df_y, df_test_x, prefix="tmp/Iteration1_"):
    """Save inputs to local pkl file."""
    print('SAVE PREFIX USED: ', prefix)
    pickle.dump(df_x, open(prefix + 'df_x.pkl', 'wb'))
    pickle.dump(df_y, open(prefix + 'df_y.pkl', 'wb'))
    pickle.dump(df_test_x, open(prefix + 'df_test_x.pkl', 'wb'))
    return


def sam_pickle_load(prefix='tmp/Iteration1_'):
    """Load pkl objects."""
    print('LOAD PREFIX USED: ', prefix)
    df_x = pickle.load(open(prefix + 'df_x.pkl', 'rb'))
    df_y = pickle.load(open(prefix + 'df_y.pkl', 'rb'))
    df_test_x = pickle.load(open(prefix + 'df_test_x.pkl', 'rb'))
    return df_x, df_y, df_test_x


def df_check_stats(*dfs):
    """To print DataFrames/ND Arrary Shape & Cols details."""
    stmt = "Data Frame Shape: %1.15s TotColumns: %1.15s ObjectCols: %1.15s"
    for df in dfs:
        df_shape = str(df.shape)
        if type(df) == pd.DataFrame:
            obj_cols = len(df.dtypes[df.dtypes == '0'])
            all_cols = len(df.dtypes)
            print(stmt % (df_shape, all_cols, obj_cols))
        elif type(df) == np.ndarray:
            size = df.shape[0]
            print('Numpy Array Size:', size)
    return


def data_transformations(raw_x, raw_y, raw_test_x, pickle_path=False):
    """Label Encodes the given data.

    If `pickle_path` provided, transformers labels will be saved as pickles
        at provided location.
    """
    d = defaultdict(preprocessing.LabelEncoder)

    # Labels Fit
    sam = pd.concat([raw_x, raw_test_x]).apply(lambda x: d[x.name].fit(x))

    # Labels Transform - Training Data
    new_x = raw_x.copy().apply(lambda x: d[x.name].transform(x))
    new_test_x = raw_test_x.copy().apply(lambda x: d[x.name].transform(x))

    le = preprocessing.LabelEncoder().fit(raw_y[u'status_group'])
    y = le.transform(raw_y[u'status_group'])

    if pickle_path:
        if pickle_path in [True, False]:
            fp1 = open('d.pkl', 'wb')
            fp2 = open('le.pkl', 'wb')
        else:
            fp1 = open(pickle_path + 'd.pkl', 'wb')
            fp2 = open(pickle_path + 'le.pkl', 'wb')
        pickle.dump(d, fp1)
        pickle.dump(le, fp2)

    return new_x, y, new_test_x


def check_metric(y_pred, y_test, show_cm=False):
    """Check score and print output."""
    if show_cm:
        print('------------------------------------------------')
        print(sk_metrics.classification_report(y_pred, y_test))
    print('------------------------------------------------')
    print('AC Score:', sk_metrics.accuracy_score(y_pred, y_test),
          'F1 Score:', sk_metrics.f1_score(y_pred, y_test, average='weighted'))
    return


def game(x_train, x_test, y_train, y_test, algo='rf', show_train_scores=True):
    """Standard Alogrithms fit and return scores.

    * Default Random State is set as 192 when posible.
    * Available models - rf, gb, knn, mc_ovo_rf, mc_ova_rf
    """
    if algo is 'rf':
        clf = RandomForestClassifier(n_jobs=-1, random_state=192)

    elif algo is 'gb':
        clf = GradientBoostingClassifier(random_state=192)

    elif algo is 'knn':
        clf = KNeighborsClassifier()

    elif algo is 'mc_ovo_rf':
        clf = OneVsOneClassifier(RandomForestClassifier(n_jobs=-1,
                                                        random_state=192))

    elif algo is 'mc_ova_rf':
        clf = OneVsRestClassifier(RandomForestClassifier(n_jobs=-1,
                                                         random_state=192))

    else:
        print('improper model name, select[rf, gb, knn, mc_ovo_rf, mc_ova_rf]')

    clf = clf.fit(x_train, y_train)

    if show_train_scores:
        check_metric(clf.predict(x_train), y_train)
    check_metric(clf.predict(x_test), y_test)
    return clf
