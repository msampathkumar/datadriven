"""Data Analysis for DataFrames."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# settings
plot_col_value_count_limit = 55
x_plots_limit = 5
y_plots_limit = 5
show_percentages = True


def sam_dataframe_cols_value_count_analysis(dataframe):
    """Value count analysis of dataframe.

    Args:
        * dataframe(pandas.Dataframe): input data

    """
    # input validations
    if type(dataframe) is not pd.core.frame.DataFrame:
        print('dataframe is not pd.core.frame.DataFrame')
        return

    # preparing a list cols
    i = 1
    bag = []
    for col in dataframe.columns:
        tmp = dataframe[col].value_counts()
        if len(tmp) < plot_col_value_count_limit:
            bag.append(col)

    # subplot
    f, axarr = plt.subplots(nrows=x_plots_limit,
                            ncols=y_plots_limit,
                            squeeze=False,
                            figsize=(x_plots_limit * 4, y_plots_limit * 4))

    # subplot - col mapper
    def hack(col_name='scheme_management', ax=axarr, fontsize=6):
        ss = dataframe[col_name].value_counts()
        if show_percentages:
            ss = 100 * ss / np.sum(ss)
        _ = ss.plot(ax=ax, kind='barh')
        ax.set_title(col_name.upper())

    # subplot - col mapping
    i = 0
    for x in range(x_plots_limit):
        for y in range(y_plots_limit):
            if i < len(bag):
                print((i + 1, col, len(bag)))
                col = bag[i]
                i += 1
                hack(col, axarr[x, y])
            else:
                # empty plots
                pass

    # Debug
    print(('Showing Plot for Columns:\n', bag))


def sam_dataframe_markup_value_counts(dataframe,
                                      max_print_value_counts=30,
                                      show_plots=False,
                                      figsize=(9, 3)):
    """Print value counts of each feature in data frame.

    plots will be invidual.
    """
    if not figsize:
        figsize = (9, 3)
    mydf = pd.DataFrame.copy(dataframe)
    i = 0
    raw_markup_data = []
    pp = raw_markup_data.append
    pp('''|Col ID|Col Name|UniqCount|Col Values|UniqValCount|''')
    pp('''|------|--------|---------|----------|------------|''')
    for col in mydf.dtypes.index:
        i += 1
        sam = mydf[col]
        sam_value_counts = sam.value_counts()
        tmp = len(sam_value_counts)
        sam_value_counts_len = len(sam_value_counts)
        if 1 < sam_value_counts_len < max_print_value_counts:
            flag = True
            for key, val in list(dict(sam_value_counts).items()):
                if flag:
                    pp('|%i|%s|%i|%s|%s|' % (i, col, sam_value_counts_len,
                                             key, val))
                    flag = False
                else:
                    pp('||-|-|%s|%s|' % (key, val))
            if show_plots:
                plt.figure(i)
                ax = sam_value_counts.plot(kind='barh', figsize=figsize)
                _ = plt.title(col.upper())
                _ = plt.xlabel('counts')
        else:
            pp('|%i|%s|%i|||' % (i, col, sam_value_counts_len))
    return raw_markup_data
