from socket import *
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


#warnings.simplefilter(action='ignore', category=FutureWarning)
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
while True:
    data = conn.recv(1024) #how many bytes of data will the server receive
    length = len(data) - 1
    new_data = data[0:length]
    var = str(data, "utf-8")
    text=repr(var)
    words = split_into_words(var)
    print(words)
    print()
    mcv=words[1]
    alkphos=words[2]
    sgpt=words[3]
    sgot=words[4]
    gammagt=words[5]
    drinks=words[6]
    print("$$$$$$$$$$$$$$$$My var$$$$$$$$$$$$$$$$$$$$$$$$$")
    print("MCV is",mcv)
    print("alkphos is",alkphos)
    print("sgpt is",sgpt)
    print("sgot is",sgot)
    print("gammagt is",gammagt)
    print("drinks is",drinks)

    print("$$$$$$$$$$$$$$$End$$$$$$$$$$$$$$$$$$$$$")
        
    print("---------------MAchine learn start-Hyper1------------------------")
    """df = pd.read_csv('allhyper1.data')
    
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
    print("accuracy through KNN: %f" %accuracy)
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
    if prediction=="['111.0']":
        reply="hyperthyroid"
    elif prediction=="['222.0']":
        reply="T3 toxic"
    elif prediction=="['333.0']":
        reply="goitre"
    elif prediction=="['444.0']":
        reply="secondary toxic"
    elif prediction=="['555']":
        reply="negative"
    print("For Hyper ans is----->> ",reply)

    conn.send(reply2.encode('ascii'))
    print ("Message sent")"""
conn.close()
