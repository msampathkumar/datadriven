# Machine Learning Engineer Nanodegree

## Capstone Project

Sampath Kumar

January 6th, 2017

__ INDEX __

[TOC]









## Definition

### Project Overview

Across Africa, cholera, typhoid, dysentery and other diseases kill thousands each year. To help the people of Tanzania(2007), The Tanzanian government, with support from UN Development Programme(UNDP), responded to the water problems by the installation of Drinking Water Taps and Decentralized the maintenance for a quick response. Today this water infrastructure is facing repair and maintenance issues causing a disconnection for drinking water needs.

The Taarifa Platform is an open source web API, designed to close citizen feedback loops. Using Taarifa people can report their social issues(like water, electricity, food and other) from different forms of communications like SMS, Web Forums, Emails or Twitter. Later these reports are placed into a workflow where they can be followed up and acted upon while engaging citizens and community. A message then will to local central governing body notifying the issue & the location.

![Image][water_pump_with_kids]

__Personal Motivation__: Long before I joined a school I remember a glimpse from the past, where lots of people standing in long queue for drinking water. May be it was the first time I have ever seen people struggling for drinking water till late evenings(like the above image show), on that day I felt sad. Every time I read about this problem statement I still do feel the same way.

We all are given wonderful opportunity(by Mother Nature/Life/God) for growth and education to live a good life filled with hard work, sweat that would lead to a happy life. And I believe its right to providing/sharing basic facilities is minimum help we can do our fellow passengers in our journey of life.

> We all are not the same but when we all share what we have, then there would be no one better than us.

### Problem Statement

Using the data gathered from Taarifa and the Tanzanian Ministry of Water, can we predict which pumps are functional, which need some repairs, and which don't work at all? Predicting one of these three classes that might have caused the failure of a water pump, can improve the maintenance operations to well prepare and ensure that clean, potable water is available to communities across Tanzania.

This project is inspired by [DataDriven][datadriven7]! From logical perspective, as we can understand we are looking at a __Supervised Multi Class Classification problem__ and our goal is identifying which label is the cause for an issues entry in Taarifa complaints records. So based on the features and size of dataset, our course of action would be to identify a classifier out of many different kinds of classifiers which could understand and predict well.

As shown in below image, we are going to do a step by step development progress on here.
![Udacity Machine Learning Course Plan][udacity_ml_course_plan]

After finding a suitable classifier we will try to build accuracy over it by featuring engineering and parameter tuning methods. At the same time we will also try to re-evaluate other suitable classifier to check if data quality is improved. As better the data quality, the better the accuracy and consistently top performing classifier will be our classifier for this classification.


### Metrics

#### Weighted F1 Score

The [F1 Score][F1_Score] can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:

```
F1 = 2 * (precision * recall) / (precision + recall)
```

When average parameter is set as `weighted`, F1 Score calculate metrics for each label, and find their average, weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.

``` python
>>> from sklearn.metrics import f1_score
>>> y_true = [0, 1, 2, 0, 1, 2]
>>> y_pred = [0, 2, 1, 0, 0, 1]
>>> f1_score(y_true, y_pred, average='weighted')
0.26...
>>> f1_score(y_true, y_pred, average=None)
array([ 0.8,  0. ,  0. ])
```

#### Accuracy Score

Accuracy classification score.

As the suggested evaluation metric of the competition, we use [Accuracy Score][accuracy_score] /Classification Rate metric.

The classification rate, which calculates the percentage of rows where the predicted class in the submission matches the actual class in the test set. The maximum is 1 and the minimum is 0. The goal is to maximize the classification rate.

<math display="block">
    <mi>Classification Rate</mi>
    <mo>=</mo>
    <msubsup><mi>(1/N)* ∑</mi>
    </msubsup>
    <msubsup><mi> I(Prediction == Actual)</mi>
    </msubsup>
</math>

</br>
Example from Python Scikit [Accuracy Score][accuracy_score]

``` python
>>> import numpy as np
>>> from sklearn.metrics import accuracy_score
>>> y_pred = [0, 2, 1, 3]
>>> y_true = [0, 1, 2, 3]
>>> accuracy_score(y_true, y_pred)
0.5
>>> accuracy_score(y_true, y_pred, normalize=False)
2
```

During the project, we will use both the metrics for measurements for understanding but as the suggested metric is Accuracy Score, we will give preference to Accuracy Score.

## Analysis

Source/data files are available at [DataDriven][datadriven7]


|File|Description|
|----|-----------|
|[Training set values][input_file1]|The independent variables for the training set|
|[Training set labels][input_file2]|The dependent variable (status_group) for each of the rows in Training set values|
|[Test set values][input_file3]|The independent variables that need predictions|
|[Submission format][input_file4]|The format for submitting your predictions|

### Data Exploration

Test & Train data sets consists of 39 columns each with 59400 rows and  14850 rows respectively, to predict 1 multi-labeled column.

Here is some simple analysis of columns along with some sample d. As the input data mostly consists of categorical data, for each we have also taken unique groups counts (or value counts) and plotted in horizontal bar charts for easy read.

__Description of the Features__

|Index|Column Name|Unique Values|Sample Data|
|-----|-----------|-------------|-----------|
|0|amount_tsh|98| ['6000', '0', '25']|
|1|date_recorded|356| ['2011-03-14', '2013-03-06', '2013-02-25']|
|2|funder|1897| ['Roman', 'Grumeti', 'Lottery']|
|3|gps_height|2428| ['1390', '1399', '686']|
|4|installer|2145| ['Roman', 'GRUMETI', 'World']|
|5|longitude|57516| ['34.9381', '34.6988', '37.4607']|
|6|latitude|57517| ['-9.85632', '-2.14747', '-3.82133']|
|7|wpt_name|37400| ['none', 'Zahanati', 'Kwa']|
|8|num_private|65| ['0', '0', '0']|
|9|basin|9| ['Lake', 'Nyasa', 'Lake']|
|10|subvillage|19287| ['Mnyusi', 'B', 'Nyamara']|
|11|region|21| ['Iringa', 'Mara', 'Manyara']|
|12|region_code|27| ['11', '20', '21']|
|13|district_code|20| ['5', '2', '4']|
|14|lga|125| ['Ludewa', 'Serengeti', 'Simanjiro']|
|15|ward|2092| ['Mundindi', 'Natta', 'Ngorika']|
|16|population|1049| ['109', '280', '250']|
|17|public_meeting|2| ['True', 'NaN', 'True']|
|18|recorded_by|1| ['GeoData', 'Consultants', 'Ltd']|
|19|scheme_management|12| ['VWC', 'Other', 'VWC']|
|20|scheme_name|2696| ['Roman', 'NaN', 'Nyumba']|
|21|permit|2| ['False', 'True', 'True']|
|22|construction_year|55| ['1999', '2010', '2009']|
|23|extraction_type|18| ['gravity', 'gravity', 'gravity']|
|24|extraction_type_group|13| ['gravity', 'gravity', 'gravity']|
|25|extraction_type_class|7| ['gravity', 'gravity', 'gravity']|
|26|management|12| ['vwc', 'wug', 'vwc']|
|27|management_group|5| ['user-group', 'user-group', 'user-group']|
|28|payment|7| ['pay', 'annually', 'never']|
|29|payment_type|7| ['annually', 'never', 'pay']|
|30|water_quality|8| ['soft', 'soft', 'soft']|
|31|quality_group|6| ['good', 'good', 'good']|
|32|quantity|5| ['enough', 'insufficient', 'enough']|
|33|quantity_group|5| ['enough', 'insufficient', 'enough']|
|34|source|10| ['spring', 'rainwater', 'harvesting']|
|35|source_type|7| ['spring', 'rainwater', 'harvesting']|
|36|source_class|3| ['groundwater', 'surface', 'surface']|
|37|waterpoint_type|7| ['communal', 'standpipe', 'communal']|
|38|waterpoint_type_group|6| ['communal', 'standpipe', 'communal']|


(These 39 column's unique values counts)(98, 356, 1897, 2428, 2145, 57516, 57517, 37400, 65, 9, 19287, 21, 27, 20, 125, 2092, 1049, 2, 1, 12, 2696, 2, 55, 18, 13, 7, 12, 5, 7, 7, 8, 6, 5, 5, 10, 7, 3, 7, 6)
(542989639101365927794152062178992491838404172880830791680000000000000(69 digits) is product of these 39 unique values, which is exponentially greater than 59K records we have) is the ideal amount of sufficient data to cover each and every category.

Input labels data has 39(27 object columns and 16 non-object columns) Features with 59,400 rows. Although we seem to have a good data set, looking at the unique values counts from below 39 columns we can say that we could potentially encounter Curse of Dimensionality. But, as we can see some of columns pairs (extraction\_type, extraction\_type\_group), (quantity & quantity\_group), (source, source\_class) seems have closer relation and column by 'recorded\_by' has only one unique value. So, we might have a chance to escape Curse of Dimensionality.


__Description of the Labels__

The labels in this dataset are simple. There are three possible values for status_group:

* functional - the water point is operational and there are no repairs needed
* functional needs repair - the water point is operational, but needs repairs
* non-functional - the water point is not operational


|PredictionLabels|Counts|Percentage|
|-----------------|------|----------|
|functional|32259|54.30|
|non functional|22824|38.42|
|functional needs repair|4317|07.26|

As numbers show we have data for unequal proportions. So the in normal circumstances if we train a model to learn the there might be changes where model tried to predict only first two groups which would only include ~92% data for learning.

To create a generic model which could work in all scenario, we will use stratification selection for splitting test-train data.

### Exploratory Visualization

Visualization of Object Columns Value Counts.

![Image][cols_value_counts]

Bar plot of all Object Column's Value counts.
__ NOTES:__ Values shown in image are log transformed to show differences visually.
![Image][features_vc_compare]

Histogram of all Object Column's Value counts.

![Image][features_vc_histogram]

__ Observations and Suggestions __

* Most of the data seems categorical: As this would increase the number of dimensions the results vary, we can take a deep look of how data is distributed across the groups and focus of the groups which contribute more information overall.

* Need to check __Date columns__
    * we shall convert date -> day, month, year, weekday, total_no_of_day_from_reference_point. These splits for two reasons.
        * Reason 1: It might be possible that in some location all specific set of complaints are registered on a start/mid/at end of the month. It might also be possible that they are registered on every Monday or so.
        * Reason 2: Taking as much information as possible.

* Need to check __Float and Bool columns__
    * Longitude & latitude(features) seem to hold (0,0) instead of NULL which is acting as outliers for now.
    * Longitude and latitude(features) are too much precise(in input date) that would make it too difficult to generalize. As generally water pumps are not installed next to each other to maintain a precision of high accuracy, we can reduce it.
    * Few boolean columns are having Nan(NAN or Null) values which does not completely make them boolean series. If we look into this scenario of observing not having data as another indication (new information), to preserve this knowledge we can convert them into labels.

* Following pairs looks closely related
    * quantity & quantity_group
    * quality_group & water_quality
    * extraction_type, extraction_type_class & extraction_type_group

* Other
    * categorical columns like installer, funder, scheme_name seems to hold data issue like text capitalization & trailing spaces.
    * recorded_by, seems to hold only a single value
    * population & amount_tsh, values are for some given as zero



### Algorithms and Techniques

As described in the introduction, a smart understanding of which water points will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

A Classifier comparison from [Sklearn's documentation][classifier_comparision_page]

As we have already taken __Random Forest Classifier__ for generating a Benchmark Score, we will use continue to use familiar (inherently) multi-class Supervised Classifiers like Tree Algorithms(__RF__, __GBT__). As these models are easy to train, self-learning & self evaluation nature make them a general good technique to consider. Unlike RF, GBT Model is a boosting method, which builds on weak classifiers. The idea is to add a classifier at a time, so that the next classifier is trained to improve the already trained ensemble.

![Classifiers Comparison][classifier_comparision]

For initial understanding, as we can see from above analysis on different kinds of datasets, I find that __Nearest Neighbor__ performs better when __Random Forest__ is performing low. From dataset features, we have coordinates like longitude, latitude and pump models and other, so we believe it might even possible that similar issues can be observed in certain neighborhoods and have been reported already so Nearest Neighbor models could also perform well.

We will be use both available(in Sklearn) __Gaussian Process__, __KMeans__ clustering for unsupervised Learning exploration. As we have taken two models for exploration, we can compare one among them like we can for RF and GBT.


Given that this is a multi class classification, we will also explore One\-vs\-Rest Sklearn's MultiClassification Technique. As the data is unbalanced, we believe that a One\-vs\-Rest might perform well. In `Refinement Phase`, we will be selecting the model which performs best from earlier suggestion unsupervised models. Thus we might be improving the well performing model to next level.

To summaries, we will be using following model in following pattern

* Supervised Learning
    * Random Forest
    * Gradient Boosting
    * Nearest Neighbors
* Unsupervised Learning
    * Gaussian
    * KMeans
* Multi Class
    * One vs Rest
    * One vs One

### Benchmark Model

With simplistic data labeling and with the help of Random Forest Classifiers, we have created a benchmark submission of 0.7970 for which source code is [here][benchmark_model].










## Methodology

As explained in Problem Description, we will all those steps but in a step by step Prototype/Spiral - Software development methodology. Prototype/Spiral development methodology, in simple steps is same as well know Waterfall model but in a iterative fashion till we reach end goal(a satisfied product).

Here is an image which would explain

![Image](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Software_development_methodologies.jpg/800px-Software_development_methodologies.jpg)

In Water Fall, as you can is like a stair case at each step we get closer to destination but once a step is passed we don't go back. So, to improve this model a cycle is added to make it look like iterative model. As each cycle complete, we fine tune the software product and we get closer to our final product.


```
Stage 1: Data Preparation ===> Stage 2: Model Building ===> Stage 3: Performance Evaluation
/\                                                                      ||
||                                                                      ||
\- ==================================================================== -/
                     (stage 3 to stage 1)
```

Not a complete or exact pattern but we follow(ed) above simple step in each iteration of data modeling. Below is our clear strategy for developing the model.

__Iteration 0__: (Benchmark)

* Goal: Set Benchmark

__Iteration 1__:

* Goal: Algorithm Selection

__Iteration 2.1__: (Data Preprocessing)

* Goal: Data Quality Improvement 1
* Object Columns fine tuning

__Iteration 2.2__: (Data Preprocessing)

* Goal: Data Quality Improvement 2
* KBest Feature Selection
* PCA Transformation

__Iteration 3__: (Refinement)

* Goal: MultiClass Algorithms and Submission

__Iteration 4__: (Refinement)

* Goal: Parameter Tuning and grid CV

__Iteration 5__: (Results)

* Goal: Final Model


__Note__: As per strategy, we are first doing Algorithm selection so we are interchanging Implementation stage and Data Processing positions.

### Implementation

In Iteration 0, our goal was to set a Benchmark Score. Here we did simple data transformations enough to train a data model and trained a Random Forest Model and got a testing score of 0.7970. We have Random Forest as our Data Model for two reason, one is that they are generally they are good fit for all kind of models(versatility) and second reason is that they generally don't need much to be fine tuned like GBT and other models.

In Iteration 1, our goal was to identify which data model is able to relate labels and data well. As each algorithms/data model is designed to identify specific way of solving problems, we need to identify the models. For Stage 1, we have used the same data transformation used in Iteration 0 for benchmark model and in model selection, no parameters were tuned than their respective defaults.

```
---------------------------------------------------------------------------
GradientBoostingClassifier(criterion='friedman_mse', init=None,
              learning_rate=0.1, loss='deviance', max_depth=3,
              max_features=None, max_leaf_nodes=None,
              min_impurity_split=1e-07, min_samples_leaf=1,
              min_samples_split=2, min_weight_fraction_leaf=0.0,
              n_estimators=100, presort='auto', random_state=192,
              subsample=1.0, verbose=0, warm_start=False)
---------------------------------------------------------------------------
Training Score
---------------------------------------------------------------------------
             precision    recall  f1-score   support

          0       0.91      0.71      0.80     31134
          1       0.12      0.69      0.21       575
          2       0.61      0.81      0.69     12841

avg / total       0.81      0.74      0.76     44550

---------------------------------------------------------------------------
AC Score: 0.735847362514
F1 Score: 0.758420128558
---------------------------------------------------------------------------
Testing Score
---------------------------------------------------------------------------
             precision    recall  f1-score   support

          0       0.90      0.70      0.79     10345
          1       0.11      0.64      0.19       185
          2       0.60      0.79      0.68      4320

avg / total       0.80      0.73      0.75     14850

---------------------------------------------------------------------------
AC Score: 0.728754208754
F1 Score: 0.752040797997
---------------------------------------------------------------------------
---------------------------------------------------------------------------
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')
---------------------------------------------------------------------------
Training Score
---------------------------------------------------------------------------
             precision    recall  f1-score   support

          0       0.73      0.60      0.66     29625
          1       0.09      0.23      0.13      1221
          2       0.40      0.50      0.45     13704

avg / total       0.61      0.56      0.58     44550

---------------------------------------------------------------------------
AC Score: 0.558092031425
F1 Score: 0.57811558815
---------------------------------------------------------------------------
Testing Score
---------------------------------------------------------------------------
             precision    recall  f1-score   support

          0       0.84      0.69      0.76      9795
          1       0.21      0.56      0.30       402
          2       0.58      0.71      0.64      4653

avg / total       0.74      0.70      0.71     14850

---------------------------------------------------------------------------
AC Score: 0.696296296296
F1 Score: 0.711138058952
---------------------------------------------------------------------------
---------------------------------------------------------------------------
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=False, random_state=192,
            verbose=0, warm_start=False)
---------------------------------------------------------------------------
Training Score
---------------------------------------------------------------------------
             precision    recall  f1-score   support

          0       1.00      0.98      0.99     24624
          1       0.93      0.98      0.96      3069
          2       0.98      0.99      0.99     16857

avg / total       0.99      0.99      0.99     44550

---------------------------------------------------------------------------
AC Score: 0.985521885522
F1 Score: 0.985581500224
---------------------------------------------------------------------------
Testing Score
---------------------------------------------------------------------------
             precision    recall  f1-score   support

          0       0.88      0.78      0.83      9087
          1       0.27      0.49      0.35       599
          2       0.73      0.81      0.77      5164

avg / total       0.80      0.78      0.79     14850

---------------------------------------------------------------------------
AC Score: 0.778181818182
F1 Score: 0.786838161735
---------------------------------------------------------------------------
```

__Note__: To make results as much reproducible as shown, we have set random_state value as 192 for possible algorithms and numpy random seed as 69572.

__Conclusion__: As we can see there are two models that are performing well(GBT, RF). Based on the testing and training data score we can say Random Forest, which seem to over fit the data and generate better results while GBT seem to fit the data properly as training and testing scores seem to be closer. Unfortunately as KNN Supervised Learning models sees to be a good fit with closely matching testing and training, but generated lower score compared to other two models.

Although it look like RF over fitting the data and making prediction from it, our goal was create a model that predicts well. As initially proposed we will considering RF for model building and GBT is know how much would a normal generalized model performance.

### Data Preprocessing

As per the strategy, we are taking two iterations just to improve data quality. As its popular known that data quality is what takes more than 50% of model building stuff, so we taken two iterations.

Below are the Iteration 2.1 changes.

__Iteration 2.1__: (Data Preprocessing)

* Goal: Data Quality Improvement 1
* Object Columns fine tuning


As mentioned earlier majority of the data is object columns,

* __Date Columns__: We have one columns _date\_recorded_, which supposed to show on which data record was added. As explained in the exploratory sections, we have created extra columns each for year, month, epoch days, date of month.

* __Boolean Columns__: Instead of deleting null or replacing null with True or False, we have converted all these into numbers and thus not losing any information.

```
True -> 1, False ->2, Nan or None ->3
```

* __Int(Float) Columns__: For some integer columns like Geo location co ordinates, the precision is so good that we can point the location to centimeter level. But this seemed to too precise(too much information) and so in hit and trial, when we reduced the precision up to 3 digit we found that our benchmark model was performing well.

For other numerical model, labeling will work as MinMaxScalar.

* __Object Columns__:
During the sub group plotting we have noticed that minor text capitalization issue and spaces. So we have applied a transformer to convert all the object data to lower case ASCII strings.

In the IPython Notebook, we have created a generic helper script to do this and IPython Widget for experimentations.

For further references say, lets call this new formatted/transformed data as PreProcessed Data. At this stage we are having 43 features.

Top 5 columns with huge varieties

* funder, 1898
* installer, 2146
* wpt_name, 37400
* subvillage, 19288
* scheme_name, 2697

For these columns as we look into details we have observed that most of the data has lots of cardinality issues and here are some stats collected for these columns.

* funder:
    * 100.0 percentage of DATA coverage mean, 1881 (in number) groups
    * 97.0 percentage of DATA coverage mean, 592 (in number) groups ##
    * 90.5 percentage of DATA coverage mean, 237 (in number) groups

* installer:
    * 100.0 percentage of DATA coverage mean, 1867 (in number) groups
    * 97.0 percentage of DATA coverage mean, 599 (in number) groups ##

* wpt_name:
    * 80.0 percentage of DATA coverage mean, 24838 (in number) groups ##

* subvillage:
    * 80.5 percentage of DATA coverage mean, 8715 (in number) groups ##
    * 83.0 percentage of DATA coverage mean, 9458 (in number) groups

* ward:
    * 80.0 percentage of DATA coverage mean, 998 (in number) groups ##
    * 91.5 percentage of DATA coverage mean, 1397 (in number) groups
    * 100.0 percentage of DATA coverage mean, 2093 (in number) groups

* scheme_name:
    * 100.0 percentage of DATA coverage mean, 2486 (in number) groups
    * 91.5 percentage of DATA coverage mean, 870 (in number) groups
    * 80.5 percentage of DATA coverage mean, 363 (in number) groups
    * 85.0 percentage of DATA coverage mean, 524 (in number) groups ##

__NOTE:__ Marked with double hashes are the selected values for coverage.

### Feature Selection

__Iteration 2.2__: (Data Preprocessing)

* Goal: Data Quality Improvement 2
* KBest Feature Selection
* PCA Transformation

After preprocessing, we have tried 3 methods of dimensionality reductions.

#### Variance Threshold:

VarianceThreshold is a simple baseline approach to feature selection. It removes all features whose variance doesn’t meet some threshold. By default, it removes all zero-variance features, i.e. features that have the same value in all samples.

We have taken a variance threshold limit of 80%, implies columns with less than 80 are to be dropped. We found one columns _recorded_by_ which has a variance threshold less than 80%.


#### KBest select
KBest is one of the Univariate feature selection methods that works by selecting the best features based on univariate statistical tests.

Well known statistical tests for classification are chi2, f_classif, mutual_info_classif. Using __Random Forest__ we are checking which statistical method is generating better results for K=30(selected columns).


|Statistical Test | Train Score| Test Score|
|-----------------|------------|-----------|
|chi2|0.98428731762065091|0.79966329966329963|
|f_classif|0.97432098765432096|0.79286195286195282|
|mutual_info_classif|0.98410774410774415|0.79447811447811445|

KBest statistical method: __Chi2__ wins.

Like we have tried all these three methods to be sure, we shall also check the number of columns to find the best number of minimum required columns/features to better score.

AMOUNT_TSH, DATE_RECORDED, FUNDER, GPS_HEIGHT, INSTALLER, LONGITUDE, LATITUDE, NUM_PRIVATE, BASIN, SUBVILLAGE, REGION, REGION_CODE, DISTRICT_CODE, LGA, WARD, POPULATION, PUBLIC_MEETING, SCHEME_MANAGEMENT, SCHEME_NAME, PERMIT, CONSTRUCTION_YEAR, EXTRACTION_TYPE, EXTRACTION_TYPE_GROUP, EXTRACTION_TYPE_CLASS, MANAGEMENT, MANAGEMENT_GROUP, PAYMENT, PAYMENT_TYPE

Results of previous runs
Trail 1

```python
[{'cols': 1, 'test': 0.52659932659932662, 'train': 0.57483726150392822},
 {'cols': 5, 'test': 0.68962962962962959, 'train': 0.94240179573512906},
 {'cols': 9, 'test': 0.7211447811447812, 'train': 0.97638608305274976},
 {'cols': 13, 'test': 0.75380471380471381, 'train': 0.97955106621773291},
 {'cols': 17, 'test': 0.76134680134680133, 'train': 0.98071829405162736},
 {'cols': 21, 'test': 0.76511784511784509, 'train': 0.98076318742985413},
 {'cols': 25, 'test': 0.80033670033670035, 'train': 0.98316498316498313},
 {'cols': 29, 'test': 0.80053872053872055, 'train': 0.98379349046015707},
 {'cols': 33, 'test': 0.80040404040404045, 'train': 0.98390572390572395},
 {'cols': 37, 'test': 0.79993265993265994, 'train': 0.98341189674523011}]
```

Trail 2
```python
[{'cols': 23, 'test': 0.7976430976430976, 'train': 0.9836812570145903},
 {'cols': 25, 'test': 0.80033670033670035, 'train': 0.98316498316498313},
 {'cols': 27, 'test': 0.80101010101010106, 'train': 0.9829405162738496},
 {'cols': 29, 'test': 0.80053872053872055, 'train': 0.98379349046015707},
 {'cols': 31, 'test': 0.80000000000000004, 'train': 0.98381593714927051}]
```

Trail 3
```python
[{'cols': 26, 'test': 0.80309764309764309, 'train': 0.98359147025813698},
 {'cols': 27, 'test': 0.80101010101010106, 'train': 0.9829405162738496},
 {'cols': 28, 'test': 0.80222222222222217, 'train': 0.98334455667789}]
```

As per Okham Razor's rules, we are going to select the simplest and well performing. Luckily, we have got kbest_selected_cols at _26_ which is comparatively top performer among other K-selections and also lower than actually number of columns

_Conclusion: Using __Chi2__ with __KBest__, we found 26 best selected columns for generating results._


#### PCA
PCA, Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.

Here is the cumulative

Like KBest, in a similar fashion we have tried PCA model but we have encounter some decrease in score. As we can understand from the results, that transformed will have lower dimensions but it might be always to learn from it.




### Refinement

__Iteration 3__: (Refinement)

* Goal: MultiClass Algorithms and Submission

#### Multi Class

From sklearn.multiclass module we have selected two simple meta-estimator One-Vs-The-Rest and One-Vs-One as they seem to easily to use and performance well.

One-Vs-The-Rest

This strategy, also known as one-vs-all, is implemented in OneVsRestClassifier. The strategy consists in fitting one classifier per class. For each classifier, the class is fitted against all the other classes. In addition to its computational efficiency (only n_classes classifiers are needed), one advantage of this approach is its interoperability. Since each class is represented by one and only one classifier, it is possible to gain knowledge about the class by inspecting its corresponding classifier. This is the most commonly used strategy and is a fair default choice.


One-Vs-One

OneVsOneClassifier constructs one classifier per pair of classes. At prediction time, the class which received the most votes is selected. In the event of a tie (among two classes with an equal number of votes), it selects the class with the highest aggregate classification confidence by summing over the pair-wise classification confidence levels computed by the underlying binary classifiers.


Results
```
---------------------------------------------------------------------------
OneVsOneClassifier(estimator=GradientBoostingClassifier(random_state=192))
---------------------------------------------------------------------------
Training Score
------------------------------------------------
             precision    recall  f1-score   support
          0       0.91      0.70      0.79     31449
          1       0.12      0.68      0.21       574
          2       0.59      0.81      0.68     12527
avg / total       0.81      0.73      0.76     44550
------------------------------------------------
AC Score: 0.732435465769
F1 Score: 0.75580830821
---------------------------------------------------------------------------
Testing Score
------------------------------------------------
             precision    recall  f1-score   support
          0       0.91      0.70      0.79     10480
          1       0.10      0.59      0.16       174
          2       0.59      0.80      0.67      4196
avg / total       0.81      0.73      0.75     14850
------------------------------------------------
AC Score: 0.725185185185
F1 Score: 0.750120842313
---------------------------------------------------------------------------
OneVsRestClassifier(estimator=GradientBoostingClassifier(random_state=192))
---------------------------------------------------------------------------
Training Score
------------------------------------------------
             precision    recall  f1-score   support
          0       0.92      0.70      0.79     31606
          1       0.10      0.74      0.18       438
          2       0.59      0.81      0.68     12506
avg / total       0.82      0.73      0.76     44550
------------------------------------------------
AC Score: 0.7311335578
F1 Score: 0.756322596966
---------------------------------------------------------------------------
Testing Score
------------------------------------------------
             precision    recall  f1-score   support
          0       0.91      0.70      0.79     10540
          1       0.09      0.70      0.15       132
          2       0.58      0.79      0.67      4178
avg / total       0.81      0.72      0.75     14850
------------------------------------------------
AC Score: 0.72404040404
F1 Score: 0.750366996656
---------------------------------------------------------------------------
OneVsOneClassifier(estimator=RandomForestClassifier(n_jobs=-1,random_state=192))
---------------------------------------------------------------------------
Training Score
------------------------------------------------
             precision    recall  f1-score   support
          0       1.00      0.97      0.98     24779
          1       0.88      0.99      0.93      2893
          2       0.98      0.99      0.98     16878
avg / total       0.98      0.98      0.98     44550
------------------------------------------------
AC Score: 0.980583613917
F1 Score: 0.98079116669
---------------------------------------------------------------------------
Testing Score
------------------------------------------------
             precision    recall  f1-score   support
          0       0.89      0.77      0.83      9238
          1       0.25      0.56      0.34       472
          2       0.73      0.81      0.77      5140
avg / total       0.81      0.78      0.79     14850
------------------------------------------------
AC Score: 0.779259259259
F1 Score: 0.790187725313
---------------------------------------------------------------------------
OneVsRestClassifier(estimator=RandomForestClassifier(random_state=192))
---------------------------------------------------------------------------
Training Score
------------------------------------------------
             precision    recall  f1-score   support
          0       0.99      0.99      0.99     24260
          1       0.96      0.99      0.97      3140
          2       0.99      0.99      0.99     17150
avg / total       0.99      0.99      0.99     44550
------------------------------------------------
AC Score: 0.990819304153
F1 Score: 0.990841995406
---------------------------------------------------------------------------
Testing Score
------------------------------------------------
             precision    recall  f1-score   support
          0       0.84      0.80      0.82      8509
          1       0.30      0.52      0.38       625
          2       0.78      0.78      0.78      5716
avg / total       0.79      0.78      0.78     14850
------------------------------------------------
AC Score: 0.778047138047
F1 Score: 0.784773818155
```

* Performance of OnevsOne with OnevsRest: As we can see the Testing Results of for both the classifier, OneVsOne performed slightly better than OneVsRest and both have results improved from standard RF and GBT classifiers.
* Performance of RF with GBT: Of course as we already knew from Algorithm selection, RF forest performed well.


__Iteration 4__: (Refinement)

* Goal: Parameter Tuning and grid CV

From Iteration 3 we have got the results of OnevsOne wrapper over Random Forest but, till now we have not yet fine tune our RF Model. Here in this iteration our goal was to fine tune RF and Grid check the results.

A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.

__Parameters__:

* n_estimators : integer, optional (default=10). The number of trees in the forest.
* criterion : string, optional (default="gini"). The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain. Note: this parameter is tree-specific.
* class_weight : dict, list of dicts, "balanced", "balanced_subsample" or None, optional (default=None) Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
The "balanced" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as
`n_samples / (n_classes * np.bincount(y))`
* max_features : int, float, string or None, optional (default="auto"). The number of features to consider when looking for the best split.
* random_state: int, random_state is the seed used by the random number generator; To make results reproducible.

``` Python
parameters = {
    'n_estimators': [10, 50, 100, 150, 200],
    'class_weight': ['balanced_subsample', 'balanced'],
    'criterion': ['gini', 'entropy'],
    # 'max_features': ['log2', 'auto', 25],
    'random_state': [192]
}
```

Best Results are generated are at following conditions

```
clf_rf = RandomForestClassifier(n_estimators=150, criterion='entropy', class_weight="balanced_subsample", n_jobs=-1, random_state=192)

AC Score: 0.81346801346801345
```

## Results

__Iteration 5:

* Final Stage

In the final stage, where we clubbed the best of Iteration 4 and Iteration 5, as expected we have received a better model which improved a bit more.

```
---------------------------------------------------------------------------
             precision    recall  f1-score   support
          0       0.89      0.80      0.84      9043
          1       0.33      0.55      0.41       659
          2       0.76      0.84      0.80      5148
avg / total       0.82      0.80      0.81     14850

---------------------------------------------------------------------------
AC Score: 0.800336700337
F1 Score: 0.807207852789
```

Final post submission score we achieved is [0.8201][final_benchmark_model].


## Conclusion
### Free-Form Visualization
### Reflection
### Improvement
* Like correlation for numerical columns, objects Columns can be looked up for Associations Techniques.
* Unsupervised learning explorations were failed and could not generate much results till now, so there is score for further research in Unsupervised learning.

## Sources & References

* [DataDriven](https://www.drivendata.org/competitions/7)
* [Choosing a ML Classifier](http://blog.echen.me/2011/04/27/choosing-a-machine-learning-classifier/)
* [Submission Code](https://github.com/msampathkumar/datadriven_pumpit)
* [Wikipedia: Water Supply & Sanitation in Tanzania](https://en.wikipedia.org/wiki/Water_supply_and_sanitation_in_Tanzania)
* [UN Report](http://www.unwater.org/fileadmin/user_upload/unwater_new/docs/Publications/TZA_pagebypage.pdf)
* [UN 2007 Water Taps Installation](http://www.un.org/africarenewal/magazine/april-2007/water-betters-lives-tanzania)
* [GBT Video Lecture](http://videolectures.net/solomon_caruana_wslmw/)
* [GBT](http://fastml.com/what-is-better-gradient-boosted-trees-or-random-forest/)
* [Classifier Comparison](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
* [Multi-class Classification](http://www.mit.edu/~9.520/spring09/Classes/multiclass.pdf)
* [Multi-class and multi label algorithms](http://scikit-learn.org/stable/modules/multiclass.html#ovr-classification)
* [Multi-class Metric](http://sebastianraschka.com/faq/docs/multiclass-metric.html)
* [Standford UnSupervised Learning](http://ufldl.stanford.edu/wiki/index.php/UFLDL_Tutorial)

<!---Input Files-->

[input_file1]: https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv
[input_file2]: https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv
[input_file3]: https://s3.amazonaws.com/drivendata/data/7/public/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv
[input_file4]: https://s3.amazonaws.com/drivendata/data/7/public/SubmissionFormat.csv
<!---Images-->

[classifier_comparision]: http://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png?style=centerme
[water_pump_with_kids]: http://drivendata.materials.s3.amazonaws.com/pumps/pumping.jpg?style=centerme
[udacity_ml_course_plan]: https://raw.githubusercontent.com/msampathkumar/datadriven_pumpit/master/images/UDACITY_ML_COURSE_GIST.png?style=centerme
[cols_value_counts]: https://raw.githubusercontent.com/msampathkumar/datadriven_pumpit/master/images/cols_value_count_li55.png?style=centerme
[features_vc_compare]: https://raw.githubusercontent.com/msampathkumar/datadriven_pumpit/master/images/features_vc_compare.png?style=centerme
[features_vc_histogram]: https://raw.githubusercontent.com/msampathkumar/datadriven_pumpit/master/images/features_vc_histogram.png?style=centerme
[new_benchmark_score]: https://raw.githubusercontent.com/msampathkumar/datadriven_pumpit/master/images/submissions_current_rank_192.png
[final_benchmark_score]: https://raw.githubusercontent.com/msampathkumar/datadriven_pumpit/master/images/BenchmarkScore_0.8201.png?style=centerme

<!---others-->

[F1_Score]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
[benchmark_model]: https://github.com/msampathkumar/datadriven_pumpit/blob/master/BenchMarkSeed_0.7970.ipynb "IPython Notebook Link"
[final_benchmark_model]: https://github.com/msampathkumar/datadriven_pumpit/blob/master/BenchMarkSeed_0.8201.ipynb "IPython Notebook Link"
[datadriven7]: https://www.drivendata.org/competitions/7/ "Driven Data Competition Page Link"
[classifier_comparision_page]: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html "Reference Page Link"
[accuracy_score]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score "Reference Page Link"

