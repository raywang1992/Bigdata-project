__author__ = 'Administrator'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pylab import *
from sklearn import tree
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydot
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data_3.txt',encoding = 'utf-8')

# print df.columns
del df['team']
del df['rival']
del df['time']
del df['score']
# print columns name
# print basic Statistics index
# including mean/stand deveation/min/25%quantile/50%quantile/75%quantile/max
# so total 1204224 records,including 602112 games
# print df.columns
print df.describe()
print df['Number of shots'].count()

# ----------------------------------3
print '---------------------------------------------------------------------------'
print 'The Pearson Analysis of factors and result:'
print "Pearson: Number of shots and result:", pearsonr(df['Number of shots'], df['result'])[0]
print "Pearson: Shots on goal and result:", pearsonr(df['Shots on goal'], df['result'])[0]
print "Pearson: Hit the post and result:", pearsonr(df['Hit the post'], df['result'])[0]
print "Pearson: pass the ball and result:", pearsonr(df['pass the ball'], df['result'])[0]
print "Pearson: Crossing and result:", pearsonr(df['Crossing'], df['result'])[0]
print "Pearson: corner kick and result:", pearsonr(df['corner kick'], df['result'])[0]
print "Pearson: offside and result:", pearsonr(df['offside'], df['result'])[0]
print "Pearson: ST(Steal) and result:", pearsonr(df['ST(Steal)'], df['result'])[0]
print "Pearson: yellow card and result:", pearsonr(df['yellow card'], df['result'])[0]
print "Pearson: red card and result:", pearsonr(df['red card'], df['result'])[0]
print "Pearson: controlling percentage and result:", pearsonr(df['controlling percentage'], df['result'])[0]


# conclusion:Shots on goal has great positive impact on score difference;red card has strong negative impact on score difference
print '---------------------------------------------------------------------------'
print 'The Pearson Analysis of factors and score difference:'
print "Pearson: Number of shots and score difference:", pearsonr(df['Number of shots'], df['score difference'])[0]
print "Pearson: Shots on goal and score difference:", pearsonr(df['Shots on goal'], df['score difference'])[0]
print "Pearson: Hit the post and score difference:", pearsonr(df['Hit the post'], df['score difference'])[0]
print "Pearson: pass the ball and score difference:", pearsonr(df['pass the ball'], df['score difference'])[0]
print "Pearson: Crossing and score difference:", pearsonr(df['Crossing'], df['score difference'])[0]
print "Pearson: corner kick and score difference:", pearsonr(df['corner kick'], df['score difference'])[0]
print "Pearson: offside and score difference:", pearsonr(df['offside'], df['score difference'])[0]
print "Pearson: ST(Steal) and score difference:", pearsonr(df['ST(Steal)'], df['score difference'])[0]
print "Pearson: yellow card and score difference:", pearsonr(df['yellow card'], df['score difference'])[0]
print "Pearson: red card and score difference:", pearsonr(df['red card'], df['score difference'])[0]
print "Pearson: controlling percentage and score difference:", pearsonr(df['controlling percentage'], df['score difference'])[0]



# plt.scatter(df['red card'], df['score difference'],s=10)
# plt.xlabel('red card')
# plt.ylabel('score difference')
# plt.show()


columns = [ 'Number of shots', 'Shots on goal', 'Hit the post', 'pass the ball', 'Crossing', 'corner kick' , 'offside', 'ST(Steal)', 'yellow card', 'red card', 'controlling percentage']
labels = df['result'].values
features = df[list(columns)].values
# print features[0]
clf = tree.DecisionTreeClassifier(criterion='entropy',min_samples_leaf=20)
clf = clf.fit(features, labels)
tree.export_graphviz(clf, out_file='tree.dot',feature_names=columns)
# tree.dot is the final result

# print type(features)
# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(features)
# print features
# print X_train_minmax
# ------------------------------------------
print '---------------------------------------------------------------------------'
print '---------------------------------------------------------------------------'
print 'Score Difference Regression Result without min_max_Normalization:'
score_differce = df['score difference'].values
lm = LinearRegression()
clf_2 = lm.fit(features, score_differce)
print pd.DataFrame(zip(columns,clf_2.coef_),columns = ['features','estimatedCoefficients'])

print '---------------------------------------------------------------------------'
print '---------------------------------------------------------------------------'
print 'Score Difference Regression Result without min_max_Normalization:'
features_1 = features.astype(np.float)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(features_1)
lm_3 = LinearRegression()
clf_3 = lm.fit(X_train_minmax, score_differce)
print pd.DataFrame(zip(columns,clf_3.coef_),columns = ['features','estimatedCoefficients'])