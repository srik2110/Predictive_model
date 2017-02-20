import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#univariate analysis
#train.dtypes
#train.describe()#for integer/numeric variables/continuous variables this gives measure of central tendency and spread of data such as mean, median, range, IQR, sd.

#Categorical variables
categorical_variables = train.dtypes.loc[train.dtypes=='object'].index
#print categorical_variables

#determine num of unique values in each column
train[categorical_variables].apply(lambda x:len(x.unique()))
#if the num of unique values is < 10, it is ok, but when the num of unique values > 10 it is generally high...

#Analyse few categories
#native_country has 42 unique values which is high.
#let us anlyse each of these categories for their counts and count%

#print train['Race'].value_counts()
#df.shape would give the shape of dataframe - rows x cols
#shape[0] - rows 
#let us get count%
#print train['Race'].value_counts()/train.shape[0]

#one can observe that top variable itself accounts for 85% and top 2 combined accounts for ~95%
#now native_country
#print train['Native.Country'].value_counts()
#print train['Native.Country'].value_counts()/train.shape[0]

#Multivariate Analysis
#there can be 3 combinations of the type of 2 variables -1) categorical-categorical2)categorical-continuous3)cont-cont

#print the cross tabulation
#ct = pd.crosstab(train['Sex'], train['Income.Group'],margins=True)
#print ct

#plot uisng a stacked chart
#ct.iloc[:-1,:-1].plot(kind = 'bar', stacked=True, color=['red','blue'], grid=False)
#plt.show()

#get in terms of percentages
def percConverter(ser):
	return ser/float(ser[-1])#divides by last cell of that row i.e All column value
#ct2 = ct.apply(percConverter, axis=1)
#ct2.iloc[:-1,:-1].plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
#plt.show()

#both continuous case
#train.plot('Age', 'Hours.Per.Week', kind='scatter')
#plt.show()

#Categorical-Continuous combination
#train.boxplot(column='Hours.Per.Week', by = 'Sex')
#plt.show()

#check missing values
#get num of missing values in each series
#print train.apply(lambda x:sum(x.isnull()))
#print test.apply(lambda x:sum(x.isnull()))

#when the missing values are categorical variables, one can impute with mode
from scipy.stats import mode
mode(train.Workclass).mode[0]

var_to_impute = ['Workclass', 'Occupation', 'Native.Country']
for var in var_to_impute:
	train[var].fillna(mode(train[var]).mode[0], inplace=True)
	test[var].fillna(mode(train[var]).mode[0], inplace=True)
#print train.apply(lambda x:sum(x.isnull()))

#outliers for numerical variables can be checked by creating scatter plots. 
#train.plot('ID', 'Age', kind='scatter')
#plt.show()

#variable transformation
categories_to_combine = ['State-gov', 'Self-emp-inc', 'Federal-gov', 'Without-pay', 'Never-worked']
for cat in categories_to_combine:
	train.Workclass.replace({cat:'Others'},inplace=True)
	test.Workclass.replace({cat:'Others'},inplace=True)

#print train.Workclass.value_counts()/train.shape[0]

categorical_variables = list(train.dtypes.loc[train.dtypes=='object'].index)
#remove workplace because already combined
categorical_variables = categorical_variables[1:]

for column in categorical_variables:
	frq = train[column].value_counts()/train.shape[0]
	categories_to_combine = frq.loc[frq.values<0.05].index

	for cat in categories_to_combine:
		train[column].replace({cat:'Others'},inplace=True)
		test[column].replace({cat:'Others'},inplace=True)

#check for unique values
#print train[categorical_variables].apply(lambda x:len(x.unique()))

#data processing
from sklearn.preprocessing import LabelEncoder
categorical_variables = train.dtypes.loc[train.dtypes=='object'].index
test_variables = categorical_variables[:-1]
#convert object type to numeric type
le = LabelEncoder()
for var in categorical_variables:
	train[var] = le.fit_transform(train[var])
for cat in test_variables:
	test[cat] = le.fit_transform(test[cat])

#print train.dtypes

from sklearn.tree import DecisionTreeClassifier

#define predictor variables except in and target
dependent_variable = 'Income.Group'
independent_variables = [x for x in train.columns if x not in ['ID', 'Income.Group']]
#print independent_variables

#initialize algorithm
model = DecisionTreeClassifier(max_depth=10, min_samples_leaf=100, max_features='sqrt')
model.fit(train[independent_variables], train[dependent_variable])

predictions_train = model.predict(train[independent_variables])
prediction_test = model.predict(test[independent_variables])

#analyze results
from sklearn.metrics import accuracy_score
acc_train = accuracy_score(train[dependent_variable], predictions_train)

#print 'Train Accuracy:%f'%acc_train
df = pd.DataFrame(prediction_test.astype(str), columns = ['C'])
df.replace(['1', '0'], ['>50', '<=50'], inplace=True)
pd.DataFrame({'ID':test['ID'], 'Income.Group':df['C']}).to_csv("sub1.csv", sep = ',', index=False)
