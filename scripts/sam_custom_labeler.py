"""A custom labler class.

Custom labler for checking and removing less used categorical
 class and also ensure to data coverage is effective.

Default settings for data coverage is 80%.
"""
import pandas as pd


class CUST_CATEGORY_LABELER():
    """Custom Mapper Function.

    Based on pd.Series.values_counts, a labler is prepared
     to cover one of following details
        1. cover top 80% of groups(DEFAULT) (or)
        2. top 500 groups

    A special `transform_analysis` function is provided to
     understand how value_counts are spread out

    Example:
        >>> # Test Data
        >>> ss = pd.Series(np.arange(5000) // 5)
        >>> ss = ss.map(lambda x: str(x))
        >>>
        >>> # creating labler
        >>> labler = CUST_CATEGORY_LABELER()
        >>> labler.fit(funder)
        >>>
        >>> # testing
        >>> _ =  labler.check_group_coverage(90)
        90 percentage of GROUPS coverage mean, 1691(in number) groups
        >>>
        >>> _ =  labler.check_data_coverage(90)
        90 percentage of DATA coverage mean, 666 (in number) groups
    """

    def __init__(self):
        """Defaults."""
        self.DB = None
        self.GROUP_COVERAGE = 500
        self.DATA_COVERAGE_LIMIT = 80
        self.DATA = None
        self.DATA_VC = None

    def fit(self, col_data):
        """Fit the data to class.

        Args:
            data(ndarray)
        """
        if type(col_data) != pd.core.series.Series:
            return 'Error: input data should be - pd.core.series.Series'

        self.DATA = col_data

        # by default values counts are sorted
        self.DATA_VC = self.DATA.value_counts()

        # converting them to percentages
        self.DATA_VC /= (self.DATA.shape[0] / 100)

    def check_data_coverage(self, data_coverage=None):
        """Check the data coverage.

        Args:
            check_data_coverage(float): Range is (0.0, 100.0)
        """
        if data_coverage is None:
            # default coverage is 80(%)
            data_coverage = self.DATA_COVERAGE_LIMIT
        if data_coverage < 1:
            return 'InputError: provide inputs between (0.0 and 100.00]'
        counter = 1
        cum_group_percentage = 0
        for group_name, group_percentage in list(self.DATA_VC.items()):
            counter += 1
            cum_group_percentage += group_percentage

            if cum_group_percentage > data_coverage:
                break
        tmp = '%s percentage of DATA coverage mean, %s (in number) groups'
        tmp %= (data_coverage, counter)
        print(tmp)
        return counter

    def check_group_coverage(self, groups_coverage=80):
        """
        param: groups_coverage - can be provided as fraction/int.

        To convert fraction into proper count for inter checks.

        Args:
            * data_coverage(int): Range between (0 - 100)
                percentage(%) of the groups to be covered.
        """
        groups_count = 0
        if groups_coverage:
            if 0 < groups_coverage < 100:
                groups_count = self.DATA_VC.shape[0] * groups_coverage / 100
            else:
                return 'InputError: input number to be in between (0.0 - 100.]'
        else:
            groups_count = self.GROUP_COVERAGE
            print('Using default groups_coverage !')
        tmp = '%s percentage of GROUPS coverage mean, %s(in number) groups'
        tmp %= (groups_coverage, groups_count)
        print(tmp)
        return groups_count

    def transform_analysis(self, data_coverage=None, groups_coverage=None):
        """Post transform data view.

        Args:
            * data_coverage(int): Range between (0 - 100)
                percentage(%) of the amount data to be covered.

            * groups_coverage(int/float):
                Limit the amount groups(variety) coverage. All input can be
                 provided as fraction or a specific count with in limit.

        Example:
            >>> labler = CUST_CATEGORY_LABELER()
            >>> labler.fit(RAW_X.funder)
            >>>
            >>> # to checking report for covering 85.50% data
            >>> labler.transform_analysis(data_coverage=85.50)
        """
        counter = 0
        cum_group_percentage = 0

        if data_coverage is None and groups_coverage is None:
            return 'No Inputs provided'

        if data_coverage or groups_coverage is None:
            # prefer coverage of data
            # # all groups
            groups_coverage = self.DATA_VC.shape[0]
        else:
            # coverage of groups
            # # all groups
            data_coverage = 100
            groups_coverage = self.check_group_coverage(groups_coverage)

        for group_name, group_percentage in list(self.DATA_VC.items()):
            counter += 1
            cum_group_percentage += group_percentage
            res = "%d, %.2f, %s, %.2f" % (counter, cum_group_percentage,
                                          group_name, group_percentage)
            print(res)

            # hard limit counter - as print used in above.
            if counter > 1000:
                break

            # soft limit counter
            if (cum_group_percentage > data_coverage):
                break
            if (counter > groups_coverage):
                break

    def transform(self, groups_coverage=None):
        """
        Default transformation is based on coverage.

        If cumulative sum of groups frequencies then
         label is to only cover upto top 80% of groups.
        """
        if not groups_coverage:
            groups_coverage = self.check_data_coverage()

        # dictionary
        self.DB = dict(self.DATA_VC.head(groups_coverage))
        ss = dict(self.DATA_VC.head(groups_coverage))

        def mapper(x):
            if x in ss:
                return x
            else:
                return 'someother'
        return self.DATA.apply(mapper)

    def fit_transform(self, col_data):
        """Fit data and then transform."""
        self.fit(col_data)
        return self.transform()

    def etransform(self, data):
        """For external pd.series transformations."""
        groups_coverage = self.check_data_coverage()

        self.DB = dict(self.DATA_VC.head(groups_coverage))
        ss = dict(self.DATA_VC.head(groups_coverage))

        def mapper(x):
            if x in ss:
                return x
            else:
                return 'someother'
        return data.apply(mapper)
