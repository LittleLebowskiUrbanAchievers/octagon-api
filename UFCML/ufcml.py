######### Base Imports #############
import pandas as pd
from ggplot import *
import matplotlib.pyplot as plt
import psycopg2
import scipy
import math
import pickle
import numpy as np
import os

################# Sklearn Imports #####################
from sklearn import datasets
# Our Classifier
from sklearn.naive_bayes import GaussianNB
# Get the accuracy score of the final model
from sklearn.metrics import accuracy_score
# Split the training set into train and test
from sklearn.model_selection import train_test_split
#MLP Classifer
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn import preprocessing
from sklearn.feature_selection import RFE


############## Paramater Switches ##############

#### NEVER TOUCH ANY OF THESE!!!!!!!!!!!!!!!!!!!!!!!!!!!!

CURRDIR = os.path.dirname(os.path.abspath(__file__))

#Scaling
NORMILIZE = True
SCALE = False

#Spliting features and classifiers
SUBSET_FEATURES = True
SUBMODELS = True

#Showing Each classifier
SEPERATE_CLASSIFIERS = True

#Combining output
NN_COMBINE = True
VOTING_CLASSIFIERS = True


def train_models():

    #Get connection to the database
    try:
        conn = psycopg2.connect("dbname='capstone' user='samkreter' host='localhost'")
    except:
        print("I am unable to connect to the database")

    cur = conn.cursor()

    #Prepare the super weird sql to pull out the features in the right order
    sql = "select   fights.f1result, \
                f1.height_inches AS f1_height,\
                f1.reach_inches AS f1_reach, \
                f1.weight_lbs AS f1_weight, \
                f1.strike_offense_per_min AS f1_strike_offense_per_min, \
                f1.strike_offense_accuracy AS f1_strike_offense_accuracy, \
                f1.strike_defense_per_min AS f1_strike_defense_per_min, \
                f1.strike_defense_accuracy AS f1_strike_defense_accuracy, \
                f1.takedowns_per_fight AS f1_takedowns_per_fight, \
                f1.takedowns_accuracy AS f1_takedowns_accuracy, \
                f1.takedowns_defense AS f1_takedowns_defense, \
                f1.submissions_per_fight AS f1_submissions_per_fight, \
                f1.total_fights AS f1_total_fights, \
                f1.association AS f1_association, \
                f1.country AS f1_country, \
                f1.wins AS f1_wins, \
                f1.losses AS f1_losses, \
                f1.total_fights AS f1_total_fights, \
                                                \
                f2.height_inches AS f2_height, \
                f2.reach_inches AS f2_reach, \
                f2.weight_lbs AS f2_weight, \
                f2.strike_offense_per_min AS f2_strike_offense_per_min, \
                f2.strike_offense_accuracy AS f2_strike_offense_accuracy, \
                f2.strike_defense_per_min AS f2_strike_defense_per_min, \
                f2.strike_defense_accuracy AS f2_strike_defense_accuracy, \
                f2.takedowns_per_fight AS f2_takedowns_per_fight, \
                f2.takedowns_accuracy AS f2_takedowns_accuracy, \
                f2.takedowns_defense AS f2_takedowns_defense, \
                f2.submissions_per_fight AS f2_submissions_per_fight, \
                f2.total_fights AS f2_total_fights, \
                f2.association AS f2_association, \
                f2.country AS f2_country, \
                f2.wins AS f2_wins, \
                f2.losses AS f2_losses, \
                f2.total_fights AS f2_total_fights \
                from octagon.fights INNER JOIN octagon.fighters AS f1 ON octagon.fights.f1name = f1.name INNER JOIN octagon.fighters AS f2 ON octagon.fights.f2name = f2.name;"

    df = pd.read_sql_query(sql, conn)

    conn.close()

    #Convert the result from text to numeric, mostly interested in the wins
    df['f1result'] = df['f1result'].replace(['win', 'draw', 'NC'],[1,2,2])


    # Take only a subset of the features found using recursive feature elimination
    if SUBSET_FEATURES:
        subset = ['f1result','f1_strike_offense_per_min','f1_strike_defense_per_min','f1_association','f1_wins','f1_losses',
            'f2_strike_offense_per_min','f2_strike_defense_per_min','f2_association','f2_wins','f2_losses']
        df = df[subset]


    ## Selection how to interpolate the missing data
    df0 = df.interpolate()
    df0 = df.dropna()
    df0 = df.interpolate(method='spline', order=2)
    df = df.interpolate(method='pchip')


    #Reverse the training data in order to train on lossing fights aswell
    f1_end = math.ceil(len(df.columns) / 2)
    df2 = df.copy()
    tmp = df2.ix[:,1:f1_end]
    df2.ix[:,1:f1_end] = df2.ix[:,f1_end:].values
    df2.ix[:,f1_end:] = tmp.values
    df2['f1result'] = 0

    #combine the dataframe into one to help with the training
    df = df.append(df2)
    df = df.sample(frac=1).reset_index(drop=True)

    #Pull the labels out of the dataset
    y = df['f1result']
    del df['f1result']

    X = df


    #
    if NORMILIZE:
        X = preprocessing.normalize(X, norm='l2')

    if SCALE:
        X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state = 1)

    names = [
             "MLP",
             "Naive Bayes",
             "KNN",
             "SVM Linear",
             "SVM gamma",
             "Decsion Tree",
             "Random Forest",
             "adaBoost"
            ]

    clfs = [
            MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1,max_iter=400),
            GaussianNB(),
            KNeighborsClassifier(3),
            SVC(kernel="linear", C=0.025,probability=True),
            SVC(gamma=2, C=1,probability=True),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
            AdaBoostClassifier()
           ]

    if SUBMODELS and SUBSET_FEATURES:
        #feats = [0,2,3]
        feats = [0,2,3,4,5,6,7]
        clfs = [clfs[i] for i in feats]
        names = [names[i] for i in feats]
    elif SUBMODELS:
        feat = [3,5,7]
        clfs = [clfs[i] for i in feats]
        names = [names[i] for i in feats]

    #Use Feature elemination to find the most important features
    if False:
        f1_end = int(len(df.columns) / 2)
        indexs = np.array([i for i in range(f1_end)])
        selector = RFE(clfs[3], 14, step=1)
        selector = selector.fit(X_train, y_train)
        print(selector.support_)
        print(indexs[selector.support_[:f1_end]])
        print(indexs[selector.support_[f1_end:]])

    #Train the individule classifiers and print their scores
    if SEPERATE_CLASSIFIERS:

        pred_list_train = []
        pred_list_test = []

        for index, clf in enumerate(clfs):
            clf.fit(X_train,y_train)
            preds = clf.predict(X_test)

            preds_train = np.amax(clf.predict_proba(X_train),axis=1)
            preds_test = np.amax(clf.predict_proba(X_test),axis=1)

            pred_list_train.append(preds_train)
            pred_list_test.append(preds_test)
            #print(names[index] + " Accuracy of the model is: %.2f%%" % (accuracy_score(preds,y_test) * 100))
            print("%.2f" % (accuracy_score(preds,y_test) * 100))


    #Use a soft committe style voting
    # This sums the class probabilites and calls and argmax for the class
    if VOTING_CLASSIFIERS:
        clf_vote = VotingClassifier(estimators=list(zip(names,clfs)),voting='soft')
        clf_vote.fit(X_train,y_train)
        preds = clf_vote.predict(X_test)
        print("%.2f" % (accuracy_score(preds,y_test) * 100))


    #Trains a MLP Neural Network on the class predicitons of each of the base networks
    if NN_COMBINE:
        combine_clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        combine_clf.fit(np.array(pred_list_train).T,y_train)
        preds = combine_clf.predict(np.array(pred_list_test).T)
        print("%.2f" % (accuracy_score(preds,y_test) * 100))



# Predict who will when in a UFC fight
# \param f1id: id for the first fighter
#\param f2id: id for the second fighter
#\return: array with probabilites of winning [fighter1,fighter2,Uncertainty]
def predict(f1id=2646,f2id=13767):


    #Open a connection to the database
    #TODO: Use env varibles for the connectoin strings
    try:
        conn = psycopg2.connect("dbname='llua' user='llua' password='capstone' host='macrohard.io'")
    except:
        print("I am unable to connect to the database")

    cur = conn.cursor()

    #Big ass sql string
    sql = "select fid, \
                height_inches, \
                reach_inches, \
                weight_lbs, \
                strike_offense_per_min, \
                strike_offense_accuracy, \
                strike_defense_per_min, \
                strike_defense_accuracy, \
                takedowns_per_fight, \
                takedowns_accuracy, \
                takedowns_defense, \
                submissions_per_fight, \
                total_fights, \
                association, \
                country, \
                wins, \
                losses, \
                total_fights FROM octagon.fighters"

    #Read all fighters in from the database.
    #TODO: find a better way for interpolation
    df = pd.read_sql_query(sql, conn)


    #convert the text fields into numeric values
    df = convert_text(df)

    #Interpolate the data to fill in the missing values
    df = df.interpolate(method='pchip')
    df = df.dropna()

    #Take a subset of the features found in the feature selection process
    if SUBSET_FEATURES:
        subset = [
                    'fid',
                    'strike_offense_per_min',
                    'strike_defense_per_min',
                    'association',
                    'wins',
                    'losses'
                    ]
        df = df[subset]

    #Pull out the fighters that we want and remove their id field
    f1 = df[df['fid'] == f1id].drop('fid',axis=1)
    f2 = df[df['fid'] == f2id].drop('fid',axis=1)


    #Error out if the fighter could not be found after data cleaning
    if f1.empty or f2.empty:
        print("Something is wrong with pulling out the stuff")
        exit(-1)


    #Create a feature vector from the two fighters data
    feat_vector = np.append(f1.values,f2.values)

    #Reshape the data to make sklearn feel better about itself and not give a warning
    feat_vector = feat_vector.reshape(1, -1)

    #Normilize the input vectors since the classifiers are trained on normilized data
    if NORMILIZE:
        feat_vector = preprocessing.normalize(feat_vector, norm='l2')


    #Load the saved classifiers
    with open(CURRDIR + "/main-clfs.pickle","rb") as f:
        clfs = pickle.load(f)


    #Names for each of the classifiers
    names = [
         "MLP",
         "KNN",
         "SVM Linear",
         "SVM gamma",
         "Voting"
        ]


    #TODO: take into account the individule values
    for index,clf in enumerate(clfs):
        print(names[index],clf.predict_proba(feat_vector))

    #Return the combined voting answer
    return clfs[-1].predict_proba(feat_vector)[0]


# Convert text fields into a numeric in the interval [0,1] based on the index
# \param df: dataframe that should be converted
# \return: the dataframe with the converted values
def convert_text(df):


    #Grab the recored names for the percentages
    with open(CURRDIR + '/text_based_names.pickle','rb') as f:
        names = pickle.load(f)

    #The preprocessed text fields
    text_based = ['country','association']

    #Iterate through and convert the text fields
    for index, name in enumerate(text_based):
        df[name] = df[name].apply(lambda x: names[index].index(x) / len(names[index]))


    return df




