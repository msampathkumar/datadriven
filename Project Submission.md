# Machine Learning Engineer Nanodegree
## Capstone Project
Sampath Kumar
December 18st, 2016

KeyWords:


# Introduction

Across Africa, cholera, typhoid, dysentery and other diseases kill thousands each year. To help the people of Tanzania(2007), The Tanzanian government, with support from UN Development Programme(UNDP), responded to the water problems by installation of Drinking Water Taps and Decentralised the maintenance for quick response. Today this water infrastructure is facing repair and maintenance issues causing a disconnection for drinking water needs.

The Taarifa Platform is an open source web API, designed to close citizen feedback loops. Using Taarifa people can report their social issues(like water, electricity, food and other) from different forms of communications like SMS, Web Forums, Emails or Twitter. Later these reports are placed into a workflow where they can be followed up and acted up on while engaging citizens and community. A message then will to local central governing body notifying the issue & the location.

![Image](http://drivendata.materials.s3.amazonaws.com/pumps/pumping.jpg)

# Problem Statement

Using the data gathered from Taarifa and the Tanzanian Ministry of Water, can we predict which pumps are functional, which need some repairs, and which don't work at all? Predicting one of these three classes based and a smart understanding of which waterpoints will fail, can improve the maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

This project is inspired by [DataDriven][1]!

# Metrics

Metric we are going to use is Accuracy Score.

As the evaluation metric of the competition use [Accuracy Score][url_metric] /Classification Rate, we can use this metric.

The classification rate, which calculates the percentage of rows where the predicted class in the submission matches the actual class in the test set. The maximum is 1 and the minimum is 0. The goal is to maximise the classification rate.

<math display="block">
    <mi>Classification Rate</mi>
    <mo>=</mo>
    <msubsup><mi>(1/N)* ∑</mi>
                <mi>i=0</mi> <mi>N</mi>
    </msubsup>
    <msubsup><mi> I (Prediction == Actual)</mi>
    </msubsup>
</math>

Example from Python Scikit

```
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]

# calculating score
accuracy_score(y_true, y_pred)
```


# Analysis

## Data Files

Here are files provided at [DataDriven][1]


|File|Description|
|----|-----------|
[Training set values][file1]|The independent variables for the training set|
[Training set labels][file2]|The dependent variable (status_group) for each of the rows in Training set values|
[Test set values][file3]|The independent variables that need predictions|
[Submission format][file4]|The format for submitting your predictions|

## Data Exploration

Test & Train Data sets consists of 39 columns, 1 multi labled columns for prediction. Shape of data set is 59400 rows with 39 columns and test data set consists of 14850 rows with 39 columns.

Here is a simple analysis of data the columns. As the input data mostly consits of categorical data, for each we have also taken unique groups counts (or value counts) and plotted in horizantal bar charts for easy read.


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

(ALL 39 columns's unique values counts)(98, 356, 1897, 2428, 2145, 57516, 57517, 37400, 65, 9, 19287, 21, 27, 20, 125, 2092, 1049, 2, 1, 12, 2696, 2, 55, 18, 13, 7, 12, 5, 7, 7, 8, 6, 5, 5, 10, 7, 3, 7, 6)
(542989639101365927794152062178992491838404172880830791680000000000000(69 digits) is product of these 39 unique values, which is exponentially greater than 59K records we have)

Input labels data has 39(27 object columns and 16 non object columns) Features with 59,400 rows. Although we seem to have a good data set, looking at the unique values counts from below 39 columns we can say that we could potentially encounter Curse of Dimensionality. But, as we can see some of columns pairs (extraction\_type, extraction\_type\_group), (quantity & quantity\_group), (source, source\_class) seems have closer relation and column by 'recorded\_by' has only one unique value. So, we might have a chance to escape Curse of Dimensionality.


## Exploratory Visualization



![Image](https://raw.githubusercontent.com/msampathkumar/datadriven_pumpit/master/cols_value_count_li55.png)



** Description of the Labels **

The labels in this dataset are simple. There are three possible values for status_group:

* functional - the water point is operational and there are no repairs needed
* functional needs repair - the water point is operational, but needs repairs
* non functional - the water point is not operational



# Algorithms and Techniques

As described in the introduction, a smart understanding of which water points will fail can improve maintenance operations and ensure that clean, potable water is available to communities across Tanzania.

We will use familiar (inherently) multi-class Supervised Classifiers like Tree Algorithms(RF/GBT)/Support Vector Machines. These are easy to train and self learning & evaluation nature make them a general good technique. During model selection we will also explore One\-vs\-Rest Sklearn's MultiClassification Technique. As the data is unbalanced, we believe having a One\-vs\-Rest might not perform well.


## Initial Project Design

As shown in below image, we are going to do a step by step development progress on here.
![Udacity Machine Learning Course Plan][image_plan]

With Random Forest Classifier, we were able to generate a benchmark of 0.7970. So, first we will start with going to deeper understanding of Random Forest worked and what features contributed it to generate this score in training.

(Implementation Plan)

1. Questions on data
2. Feature Exploration
 * PCA Transformation Checking
 * Select K Best Checking
 * Exploration - outliers check
3. Algorithm Selection
 * Unsupervised Learning Exploration(Gaussian Process, Neural Nets)
 * Supervised Learning(GBT Trees, Nearest Neighbours, RF, One-vs-One)
 * Parameter Tuning
4. Evaluation. Back to 1 with.
5. Re-Evaluation with threshold improvisation check.
6. Submission

![Classifiers Comparison](http://scikit-learn.org/stable/_images/sphx_glr_plot_classifier_comparison_001.png)

As we can see from above analysis, I find that `Nearest Neighbour` performs better when Random Forest is performing low. Also for different learning process from that of Random Forest. GBT Tree, sometime have seems performed better than Random Forest.

We will be using Gaussian Process, Neural Nets for unsupervised Learning exploration. No specific reason but taken, two models different kinds of models for exploration.






## Benchmark Model

With a simplistic data transformation and with the help of Random Forest Classifiers, we have created a benchmark submission of 0.7970 for which source code is [here](https://github.com/msampathkumar/datadriven_pumpit/blob/master/BenchMarkSeed_0.7970.ipynb)


















References

[1]: https://www.drivendata.org/competitions/7/ "Data Driven"
[file1]: https://s3.amazonaws.com/drivendata/data/7/public/4910797b-ee55-40a7-8668-10efd5c1b960.csv
[file2]: https://s3.amazonaws.com/drivendata/data/7/public/0bf8bc6e-30d0-4c50-956a-603fc693d966.csv
[file3]: https://s3.amazonaws.com/drivendata/data/7/public/702ddfc5-68cd-4d1d-a0de-f5f566f76d91.csv
[file4]: https://s3.amazonaws.com/drivendata/data/7/public/SubmissionFormat.csv
[image_plan]: https://github.com/msampathkumar/datadriven_pumpit/blob/master/UDACITY_ML_COURSE_GIST.png?raw=true
[url_metric]: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score

