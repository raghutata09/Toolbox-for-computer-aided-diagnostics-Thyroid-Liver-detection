from socket import *
import random
import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing, neighbors
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier,VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,svm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
warnings.filterwarnings('ignore')
########################################################################################
######################################Decision Functions####################################
########################################################################################
def unique_vals(rows, col):
    return set([row[col] for row in rows])

def class_counts(rows):
    
    counts = {}  
    for row in rows:
       
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def is_numeric(value):
  
    return isinstance(value, int) or isinstance(value, float)

class Question:
    

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
       
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            header[self.column], condition, str(self.value))

def partition(rows, question):
    
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity




def info_gain(left, right, current_uncertainty):
    
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)




def find_best_split(rows):
    
    best_gain = 0  
    best_question = None  
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1 

    for col in range(n_features):  

        values = set([row[col] for row in rows])  

        for val in values:  

            question = Question(col, val)

           
            true_rows, false_rows = partition(rows, question)

           
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question



class Leaf:
   

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
   

    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    

    gain, question = find_best_split(rows)

    
    if gain == 0:
        return Leaf(rows)

  
    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)

  
    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def classify(row, node):
    

   
    if isinstance(node, Leaf):
        return node.predictions

   
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)




def print_leaf(counts):
    
    pred_list = []
    for key, value in counts.items():
        pred_list.append(key)
        #print("KEY: ", key)
    return pred_list
##################################################################
######################################  KNN Functions  ##################3
##################################################################
def spliting_k(df, ratio):
	trainSize = int(len(df) * ratio)
	trainSet = []
	df_same = list(df)
	while len(trainSet) < trainSize:
		index = random.randrange(len(df_same))
		trainSet.append(df_same.pop(index))
	return [trainSet, df_same]



#############################
# SIMILARITY CHECK FUNCTION #
#############################

# euclidean distance calcualtion

import math
def euclideanDistance(instance1, instance2, length):
        distance = 0
        for x in range(length):
                distance += pow(float(instance1[x]) - float(instance2[x]), 2)
        return math.sqrt(distance)



############################################################
# NEIGHBOURS - selecting subset with the smallest distance #
############################################################

import operator 
def getNeighbors_k(trainingSet, testInstance, k):
        distances = []
        length = len(testInstance)-1
        for x in range(len(trainingSet)):
                dist = euclideanDistance(testInstance, trainingSet[x], length)
                distances.append((trainingSet[x], dist))
                
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
                neighbors.append(distances[x][0])
        return neighbors



######################
# PREDICTED RESPONSE #
######################

import operator
def getResponse_k(neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
                response = neighbors[x][-1]
                if response in classVotes:
                        classVotes[response] += 1
                else:
                        classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]



######################
# MEASURING ACCURACY #
######################

def getAccuracy_k(testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
                if (testSet[x][-1] == predictions[x]): 
                        correct = correct + 1
        return (correct/float(len(testSet))*100) 
def main():
        # prepare data
        dataframe = pd.read_csv('allhyper1.data')
        
        dataframe =dataframe.convert_objects(convert_numeric=True)
        dataframe.replace('?',np.NaN, inplace=True)
        dataframe =dataframe.convert_objects(convert_numeric=True)
        dataframe.fillna(dataframe.mean(),inplace=True)
        dataframe.drop('id',1,inplace=True)
        dataframe.drop('age',1,inplace=True)
        dataframe.drop('tbg',1,inplace=True)
        dataframe.drop('tbg_m',1,inplace=True)
        dataframe.drop('sex',1,inplace=True)
        dataframe.drop('rs',1,inplace=True)
        dataframe = np.array(dataframe)
        train, test = spliting_k(dataframe, 0.70)
        # generate predictions
        predictions=[]
        k = 3
        for x in range(len(test)):
                neighbors = getNeighbors_k(train, test[x], k)
                result = getResponse_k(neighbors)
                predictions.append(result)
#                print('> predicted=' + repr(result) + ', actual=' + repr(float(test[x][-1])))
        accuracy = getAccuracy_k(test, predictions)
        #prediction
        #t=np.array([33,00,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0.2,1,1.5,1,157,1,0.91,1,172,0,-99999,6])
        
        print('Accuracy for knn: ' + repr(accuracy) + '%')
##################################################################
#######################NAive Bayes Functions###############################
##################################################################
#calculating the mean of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

#calculating the standard deviation
def stddev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

#Now here using the mathhematical formula using mean and stddev
def probability(x, mean, stddev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stddev,2))))
	return (1 / (math.sqrt(2*math.pi) * stddev)) * exponent

#splitting the data here by 70:30
def spliting(df, ratio):
	trainSize = int(len(df) * ratio)
	trainSet = []
	df_same = list(df)
	while len(trainSet) < trainSize:
		index = random.randrange(len(df_same))
		trainSet.append(df_same.pop(index))
	return [trainSet, df_same]

def collect(df):
	summaries = [(mean(attribute), stddev(attribute)) for attribute in zip(*df)]
	del summaries[-1]
	return summaries

def setbyClass(df):
    # separate all data row by their class like {0}:[a,b,d],{1}:[c,e]
	separated = {}
	summaries = {}
	for i in range(len(df)):
		vector = df[i]
		if (vector[-1] not in separated): #class is not present in tuple then add it to it
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
        
	for classValue, instances in separated.items():
		summaries[classValue] = collect(instances)
	return summaries

def probabilityofClass(summaries, testdata):
	probabilities = {}
	for classValue, classSummaries in summaries.items():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stddev = classSummaries[i]
			if stddev == 0.0:
			    stddev=0.00000001 #tends to zero
			x = testdata[i]
			probabilities[classValue] *= probability(x, mean, stddev)
	return probabilities
		
#predicting the label from here	
def predict(summaries, testdata):
	probabilities = probabilityofClass(summaries, testdata)
	bestLabel, bestProbability = None, -1
	for classValue, probability in probabilities.items():
		if bestLabel is None or probability > bestProbability:
			bestProbability = probability
			bestLabel = classValue
	return bestLabel

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

#accuracy calculate
def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		if testSet[i][-1] == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
##################################################################
##############################Socket Programming########################
##################################################################
from nltk.tokenize import word_tokenize
def split_into_words(text):
    try:
        lis = text.split(",")
        return lis

    except Exception as e:
        print(str(e))
print("Welcome")
HOST = "192.168.43.35" #local host
PORT = 7050 #open port 7000 for connection
print("Connecting")
s = socket(AF_INET, SOCK_STREAM)
s.bind((HOST, PORT))
print("Going to listen")
s.listen(1) #how many connections can it receive at one time
conn, addr = s.accept() #accept the connection
print ("Connected by: ", addr)  #print the address of the person connected
print("Which algorithm accuracy you want to see")
print("1. Naive Bayes")
print("2. KNN algorithm")
print("3. Decision Tree")
raghu=int(input())
# ~~~~~~~~~~~~~~~Code for Naive Bayes
if raghu==1:
    df = pd.read_csv('allhyper1.data')
    df =df.convert_objects(convert_numeric=True)
    df.replace('?',np.NaN, inplace=True)
    df =df.convert_objects(convert_numeric=True)
    df.fillna(df.mean(),inplace=True)
    df.drop('id',1,inplace=True)
    df.drop('rs',1,inplace=True)
    df.drop('tbg',1,inplace=True)
    df.drop('tbg_m',1,inplace=True)
    df.drop('age',1,inplace=True)
    df.drop('sex',1,inplace=True)
    df = np.array(df)
    train, test = spliting(df, 0.70)
    summaries = setbyClass(train)
    predictions = getPredictions(summaries, test)
    accuracy = getAccuracy(test, predictions)
    print("Accuracy for Naive Bayes is ",accuracy)
elif raghu==2:
    main()
else:
    df = pd.read_csv('allhyper1.data')
    df =df.convert_objects(convert_numeric=True)
    df.replace('?',np.NaN, inplace=True)
    df =df.convert_objects(convert_numeric=True)
    df.fillna(df.mean(),inplace=True)
    df.drop('id',1,inplace=True)
    df.drop('rs',1,inplace=True)
    df.drop('tbg',1,inplace=True)
    df.drop('tbg_m',1,inplace=True)
    df.drop('age',1,inplace=True)
    df.drop('sex',1,inplace=True)

    X= np.array(df)
    X_train, X_test = train_test_split(X, test_size = 0.2)
    training_data = np.array(X_train)

    if __name__ == '__main__':

        my_tree = build_tree(training_data)


        
        testing_data = np.array(X_test)
        correct = 0
        for row in testing_data:
            actual = row[-1]
            pred = print_leaf(classify(row, my_tree))
            #print ("acutual: ",actual, " Pred: ", pred)
            

            if actual == pred:

                correct+=1
        
        acc = (float(correct)/float(len(testing_data)))
        print ("ACCURACY for Decision Tree", acc*100.00)
        

print()
print()
print("Now going to predict the disease for hyper and hypo both")
print()
print()
##################################################################
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Actual Code starts ^^^^^^^^^^^^^^^^^^^^^^
##################################################################
while True:
    data = conn.recv(1024) #how many bytes of data will the server receive
    length = len(data) - 1
    print(data)
    new_data = data[0:length]
    var = str(data, "utf-8")
    text=repr(var)
    words = split_into_words(var)
    print(words)
    print()
    age=words[1]
    sex=words[2]
    on_thyroxine=words[3]
    query_on_thyroxine=words[4]
    on_antithyroid_medication=words[5]
    sick=words[6]
    pregnant=words[7]
    thyroid_surgery=words[8]
    I131_treatment=words[9]
    query_hypothyroid=words[10]
    query_hyperthyroid=words[11]
    lithium=words[12]
    goitre=words[13]
    tumor=words[14]
    hypopituitary=words[15]
    psych=words[16]
    TSH_measured=words[17]
    TSH=words[18]
    T3_measured=words[19]
    T3=words[20]
    TT4_measured=words[21]
    TT4=words[22]
    T4U_measured=words[23]
    T4U=words[24]
    FTI_measured=words[25]
    FTI=words[26]
    TBG_measured=words[27]
    TBG=words[28]
    referral_source=words[29]
    print("$$$$$$$$$$$$$$$$My var$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("Age is",age)
    print("SEx is",sex)
    print("on thyroxine is",on_thyroxine)
    print("query on thyroxine is",query_on_thyroxine)
    print("on antithyroid medication is",on_antithyroid_medication)
    print("sick is",sick)
    print("pregnant is",pregnant)
    print("thyroid surgery is",thyroid_surgery)
    print("I131 treatment is",I131_treatment)
    print("query hypothyroid is",query_hypothyroid)
    print("query hyperthyroid is",query_hyperthyroid)
    print("lithium is",lithium)
    print("goitre is",goitre)
    print("tumor is",tumor)
    print("hypopituitary is",hypopituitary)
    print("psych is",psych)
    print("TSH measured is",TSH_measured)
    print("TSH is",TSH)
    print("T3 measured is",T3_measured)
    print("T3 is",T3)
    print("TT4 measured is",TT4_measured)
    print("TT4 is",TT4)
    print("T4U measured is",T4U_measured)
    print("T4U is",T4U)
    print("FTI measured is",FTI_measured)
    print("FTI is",FTI)
    print("TBG measured is",TBG_measured)
    print("TBG is",TBG)
    print("referral source is",referral_source)

    print("$$$$$$$$$$$$$$$End$$$$$$$$$$$$$$$$$$$$$")
        
    print("---------------MAchine learn start-Hyper1------------------------")
    df = pd.read_csv('allhyper1.data')
    
    df.replace('?',-99999, inplace=True)
    #df.replace('?',np.NaN,inplace=True)
    
    df.drop('id',1,inplace=True)
    df.drop('age',1,inplace=True)
    df.drop('sex',1,inplace=True)
    df.drop('tbg',1,inplace=True)
    df.head()
    df =df.convert_objects(convert_numeric=True)
    #df = df.dropna(subset=['sex','age','tbg'])
    df.isnull().sum(axis=0)
    df = df.fillna(df.mean())
    df['class'] = df['class'].astype(str)
    df = df.fillna(-99999)
    X=np.array(df.drop(['class'],1))#,'age','sex','tbg'
    y = np.array(df['class'])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    a=AdaBoostClassifier(RandomForestClassifier(),n_estimators=20)
    c=BaggingClassifier(DecisionTreeClassifier(),max_samples=0.8, max_features=1.0,n_estimators=20)#BaggingClassifier(LinearDiscriminantAnalysis(n_components=4, priors=None, shrinkage=None,solver='svd', store_covariance=True, tol=0.0001),max_samples=0.8, max_features=1.0,n_estimators=20)
    d=AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10)
    clf =VotingClassifier(estimators=[('a',a),('d',d)], voting='hard')
    clf.fit(X_train, y_train)
    accuracy =clf.score(X_test, y_test)
    print("accuracy through Voting: %f" %accuracy)
    #print("accuracy through SVM: %f" %accuracy_svm)

    ############################################################svm code ends

    ##############below is our input user data
    example_measures=np.array([on_thyroxine,query_on_thyroxine,on_antithyroid_medication,sick,pregnant,thyroid_surgery,I131_treatment,query_hypothyroid,
    query_hyperthyroid,lithium,goitre,tumor,hypopituitary,psych,TSH_measured,TSH,T3_measured,T3,TT4_measured,TT4,T4U_measured,T4U,FTI_measured,FTI,TBG_measured,referral_source])
    example_measures=example_measures.reshape(1,-1)
    prediction=clf.predict(example_measures)

    
    
    #len(example_measures)

    print(prediction)
    print(type(prediction))
    prediction=str(prediction)
    print(type(prediction))
    print(prediction)
    print("---------------MAchine learn end------------------------")
    if prediction=="['111']":
        reply="hyperthyroid"
    elif prediction=="['222']":
        reply="T3 toxic"
    elif prediction=="['333']":
        reply="goitre"
    elif prediction=="['444']":
        reply="secondary toxic"
    elif prediction=="['555']":
        reply="negative"
    print("For Hyper ans is----->> ",reply)
    print("-------------------Machine starts Hypo--------------")
    df = pd.read_csv('allhypo.data')
    df.replace('?',np.NaN,inplace=True)
    df.drop('id',1,inplace=True)
    df.drop('age',1,inplace=True)
    df.drop('sex',1,inplace=True)
    df.drop('tbg',1,inplace=True)
    df.head()
    df =df.convert_objects(convert_numeric=True)
    #df = df.dropna(subset=['sex','age'])
    df.isnull().sum(axis=0)
    df = df.fillna(df.mean())
    df['class'] = df['class'].astype(str)
    df = df.fillna(-99999)
    X=np.array(df.drop(['class'],1))
    y = np.array(df['class'])
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    clf.fit(X_train, y_train)
    accuracy =clf.score(X_test, y_test)
    print("accuracy through KNN: %f" %accuracy)
    

    ##############below is our input user data
    example_measures=np.array([on_thyroxine,query_on_thyroxine,on_antithyroid_medication,sick,pregnant,thyroid_surgery,I131_treatment,query_hypothyroid,
    query_hyperthyroid,lithium,goitre,tumor,hypopituitary,psych,TSH_measured,TSH,T3_measured,T3,TT4_measured,TT4,T4U_measured,T4U,FTI_measured,FTI,TBG_measured,referral_source])
    example_measures=example_measures.reshape(1,-1)
    prediction=clf.predict(example_measures)
    x=str(prediction)
    print(x)

    if x=="['222']":
        rep="Primary hypothyroid"
    elif x=="['333']":
        rep="compensated hypothyroid"
    elif x=="['444']":
        rep="secondary hypothyroid"
    elif x=="['555']":
        rep="negative"
    print("For Hypo ans is----->> ",rep)
    print("-------------------------Machine-hypo ends----------------")
    print("-------------Final Result----------------------")
    """if rep=="negative" and reply!="negative" :
        reply2=reply
    elif rep!="negative" and reply=="negative" :
        reply2=rep
    else:
        reply2="No Thyroid"""
    reply2=reply+'  '+rep
    print("Final result is",reply2)
    conn.send(reply2.encode('ascii'))
    print ("Message sent")
conn.close()
