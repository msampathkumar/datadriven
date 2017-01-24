"""Script for all local tools."""
from __future__ import absolute_import
import sys
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


try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


from IPython.core.magic import (Magics, magics_class, cell_magic)
from IPython.core.display import HTML
# from IPython.display import display

from markdown import markdown
from datetime import datetime

now = datetime.now


@magics_class
class MarkdownMagics(Magics):
    """Markdown ipython helper tools."""

    @cell_magic
    def asmarkdown(self, line, cell):
        """To print data as markdown."""
        buffer = StringIO()
        stdout = sys.stdout
        sys.stdout = buffer
        try:
            exec(cell, locals(), self.shell.user_ns)
        except:
            sys.stdout = stdout
            raise
        sys.stdout = stdout
        ext_format = markdown(buffer.getvalue(),
                              extensions=['markdown.extensions.extra'])
        return HTML("<p>{}</p>".format(ext_format))
        return buffer.getvalue() + 'test'

    def timer_message(self, start_time):
        """To check rum time."""
#         print self
        time_diff = (now() - start_time).total_seconds()
        if time_diff < 0.001:
            time_diff = 0
            print('\n<pre>In', time_diff, 'Secs</pre>')
        else:
            print('\n<pre>In', time_diff, 'Secs</pre>')

    @cell_magic
    def timer(self, line, cell):
        """Wrapper for timer funciton."""
        now = datetime.datetime.now
        start_time = now()
        buffer = StringIO()
        stdout = sys.stdout
        sys.stdout = buffer
        try:
            exec(cell, locals(), self.shell.user_ns)
            self.timer_message(start_time)
        except:
            sys.stdout = stdout
            raise
        ext_format = markdown(buffer.getvalue(),
                              extensions=['markdown.extensions.extra'])
        return HTML("<p>{}</p>".format(ext_format))
        return buffer.getvalue() + 'test'


def show_object_dtypes(df, others=True):
    """Show dataframe  dtypes for non object cols."""
    dtype = object
    if others:
        return df.dtypes[df.dtypes == dtype]
    else:
        return df.dtypes[df.dtypes != dtype]


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
    ac_score = sk_metrics.accuracy_score(y_pred, y_test)
    f1_score = sk_metrics.f1_score(y_pred, y_test, average='weighted')
    if show_cm:
        print('------------------------------------------------')
        print(sk_metrics.classification_report(y_pred, y_test))
    print('------------------------------------------------')
    print('AC Score:', ac_score,
          'F1 Score:', f1_score)
    return (ac_score, f1_score)


def game(x_train, x_test, y_train, y_test, algo='rf', show_train_scores=False):
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
        print('improper model name, please check help')

    clf = clf.fit(x_train, y_train)

    # if user does not opt
    ac_score, f1_score = 0, 0

    if show_train_scores:
        ac_score, f1_score = check_metric(clf.predict(x_train), y_train)

    ac_score1, f1_score1 = check_metric(clf.predict(x_test), y_test)
    ret = {'classifier': clf,
           'test_ac_score': ac_score,
           'test_f1_score': f1_score,
           'train_ac_score': ac_score1,
           'train_f1_score': f1_score1,
           }
    return ret
