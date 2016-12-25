import numpy as np
import pandas as pd

class CUST_CATEGORY_LABELER():
    '''
    A Custom Mapper Function
    
    Based on pd.Series.values_counts, a labler is prepared
     to cover one of following details
        1. cover top 80% of groups(DEFAULT) (or)
        2. top 500 groups
    
    A special `transform_analysis` function is provided to
     understand how value_counts are spread out
    '''
    def __init__(self):
        self.db = None
        self.groups_coverage = 500
        self.data_coverage_limit = 80
        self.data = None
        self.data_vc = None
    
    def fit(self, col_data):
        '''
        Args:
            data(ndarray)
        '''
        if type(col_data) != pd.core.series.Series:
            return 'Error: input data should be - pd.core.series.Series'

        self.data = col_data
        
        # by default values counts are sorted
        self.data_vc = self.data.value_counts()
        
        # converting them to percentages
        self.data_vc /= (self.data.shape[0] / 100)
        
    def get_groups_count_limit(self, groups_coverage):
        '''
        param: groups_coverage - can be provided as fraction/int.
        
        To convert fraction into proper count for inter checks.
        
        Args:
            limit(int/float):
        '''
        if limit:
            if 0 < limit < 1:
                groups_coverage = self.data_vc.shape[0] * (limit * 0.01)
            elif 1 < limit < self.data_vc.shape[0]:
                groups_coverage = limit
        else:
            groups_coverage = self.groups_coverage
            print('Using default groups_coverage !')
        return groups_coverage
        
        
    def transform_analysis(self, data_coverage=None, groups_coverage=None):
        '''
        Args:
            * data_coverage(int): Range between (0 - 100)
                percentage(%) of the amount data to be covered.
                
            * groups_coverage(int/float):
                Limit the amount groups(variety) coverage. All input can provided as fraction or a specific count with in limit.
                
        Example:
            >>> labler = CUST_CATEGORY_LABELER()
            >>> labler.fit(RAW_X.funder)
            >>> 
            >>> # to checking report for covering 85.50% data
            >>> labler.transform_analysis(data_coverage=85.50)
        '''
        counter = 0
        cum_group_percentage = 0
        
        if data_coverage == groups_coverage == None:
            return 'No Inputs provided'
        
        if data_coverage or groups_coverage is null:
            # prefer coverage of data
            groups_coverage = self.data_vc.shape[0] # all groups
        else:
            # coverage of groups
            data_coverage = 100 # all data
            groups_coverage = self.get_groups_count_limit(groups_coverage)
        
        for group_name, group_percentage in self.data_vc.iteritems():
            counter += 1
            cum_group_percentage += group_percentage
            res =  "%d, %.2f, %s, %.2f" % (counter, cum_group_percentage, group_name, group_percentage)
            print(res)

            # hard limit counter - as print used in above.
            if counter > 1000:
                break
                
            # soft limit counter
            if (cum_group_percentage > data_coverage) or (counter > groups_coverage):
                break
              
    def get_coverage_groups_count(self, data_coverage=None):
        '''
        To find how many groups required to cover a major portion of data(80% default).
        '''
        if not data_coverage:
            data_coverage = self.data_coverage_limit
        counter = 1
        cum_group_percentage = 0
        for group_name, group_percentage in self.data_vc.iteritems():
            counter += 1
            cum_group_percentage += group_percentage
            
            if cum_group_percentage > data_coverage:
                break
        print('Total number of groups in top %s data: %s' % (data_coverage, counter))
        return counter

    def transform(self, groups_coverage=None):
        '''
        Default transformation is based on coverage.
        
        If cumulative sum of groups frequencies then label is to only cover upto top 80% of groups.
        '''
        if not groups_coverage:
            groups_coverage = self.get_coverage_groups_count()
            
        # dictionary
        self.db = dict(self.data_vc.head(groups_coverage))
        
        ss = self.db
        def mapper(x):
            if x in ss:
                return x
            else:
                return  'someother'
        tmp = self.data.apply(mapper)
        return tmp
    
    def fit_transform(self, col_data):
        self.fit(col_data)
        return self.transform()