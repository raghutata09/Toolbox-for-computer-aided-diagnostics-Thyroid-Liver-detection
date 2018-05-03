import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing, neighbors
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
import warnings
import time
start_time = time.clock()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings('ignore')

df = pd.read_csv('allhyper1.data')
df.replace('?',-99999, inplace=True)
#df.replace('?',np.NaN,inplace=True)
df.drop('id',1,inplace=True)
#df.head()
#df =df.convert_objects(convert_numeric=True)
#df = df.dropna(subset=['sex','age'])
#df.isnull().sum(axis=0)
#df = df.fillna(df.mean())
#df['class'] = df['class'].astype(str)
#df = df.fillna(-99999)
X=np.array(df.drop(['class'],1))#,'age','sex','tbg','tbg_m','rs'
y = np.array(df['class'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


#KNN
#clf = neighbors.KNeighborsClassifier()
#clf.fit(X_train, y_train)
#accuracy =clf.score(X_test, y_test)
"""a=AdaBoostClassifier(RandomForestClassifier(),n_estimators=20,learning_rate=1)
c=BaggingClassifier(LinearDiscriminantAnalysis(n_components=4, priors=None, shrinkage=None,solver='svd', store_covariance=True, tol=0.0001),max_samples=0.8, max_features=1.0,n_estimators=20)
d=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10,learning_rate=1)

r=VotingClassifier(estimators=[('a',a), ('c',c), ('d',d)], voting='hard')



r.fit(X_train, y_train)
accuracy =r.score(X_test, y_test)
print("accuracy is :",accuracy)
example_measures=np.array([33,00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.2,1,1.5,1,157,1,0.91,1,172,0,-99999,6])
example_measures=example_measures.reshape(1,-1)
prediction=r.predict(example_measures)
print(prediction)"""








#permutation and combination 
#r=RandomForestClassifier(n_estimators=10)
#k=neighbors.KNeighborsClassifier(n_neighbors=5)
#dt=DecisionTreeClassifier()
#lda=LinearDiscriminantAnalysis(n_components=6, priors=None, shrinkage=None,solver='svd', store_covariance=False, tol=0.0001)
a=AdaBoostClassifier(RandomForestClassifier(),n_estimators=20)
b=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.8, max_features=1.0,n_estimators=20)
c=BaggingClassifier(LinearDiscriminantAnalysis(n_components=4, priors=None, shrinkage=None,solver='svd', store_covariance=True, tol=0.0001),max_samples=0.8, max_features=1.0,n_estimators=20)
d=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10)







algorithm_comb=[]
#algorithm_comb.append(('Gaussian Naive Bayes',gnb_clf))
#algorithm_comb.append(('Multinomial Naive Bayes',mnb_clf))
#algorithm_comb.append(('Bernouli naive Bayes',bnb_clf))
#algorithm_comb.append(('Ada boost-BAgging SVM',raghu1))
#algorithm_comb.append(('K-Nearest Neighbors',k))
#algorithm_comb.append(('Decision Tree',d))
#algorithm_comb.append(('Random Forest',r))
algorithm_comb.append(('AdaBoost Random Forest',a))
algorithm_comb.append(('Bagging Classifier Decision Tree',b))
algorithm_comb.append(('Bagging LinearDiscriminantAnalysis ',c))
algorithm_comb.append(('Ada boost- Decision ',d))

vrsn=tuple(algorithm_comb)
#Voting Classifier
for x in range(1,16):
    collect = []
    for y in range(0,4):
        if x & 1 << y:
            print(y+1)
            collect.append(list(vrsn[y]))
    if len(collect) >0 :
        result= VotingClassifier(collect,voting='hard')
        result.fit(X_train, y_train)
        ans=result.score(X_test, y_test)
        print("Accuracy based on Voting is ",ans)
        print("--- %s seconds ---" % (time.clock() - start_time))
    
    del(collect)

