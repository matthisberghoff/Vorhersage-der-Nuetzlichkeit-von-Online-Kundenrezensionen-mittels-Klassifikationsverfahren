# -*- coding: utf-8 -*-

"""
WiWi Master Seminar im Sommersemester 2020:
Big Data und Analytics
Vorhersage der Nützlichkeit von Online-Kundenrezensionen mittels Klassifikationsverfahren

Matthis Berghoff
"""
#Grundwerkzeug
import numpy as np
import pandas as pd
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import textblob
import textstat 
from textblob_de import TextBlobDE as TextBlobDE
from textblob import TextBlob as TextBlobEN
import nltk
#nltk.download('punkt') #Muss beim ersten mal heruntergeladen werden
import datetime


#Datenvorbereitung und Messung
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score


#Klassifikationsalgorithmen in Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm

#Allgemeine Einstellungen
pd.set_option('display.max_columns', None)

#Importieren der deutschen Rezensionen als DataFrame
germanX = pd.read_csv("amazon_reviews_multilingual_DE_v1_00.tsv", sep = "\t")

"""Data Exploration"""
#print(germanX.head())
#print (germanX.info())
#print(germanX.describe())
#print(germanX.columns)

#Anzahl Reviews pro Produktkategorie
#germanX.groupby('product_category').count()

"""Preprocessing"""
#Entfernen der Reviews mit weniger als 10 Votes
germanX.drop(germanX[germanX['total_votes'] < 10].index, inplace = True) 

#Entfernen nicht benötigter Features/Spalten
del germanX['marketplace']
del germanX['customer_id']
del germanX['review_id']
del germanX['product_id']
del germanX['product_parent']
del germanX['product_title']

#Auswahl Produktkategorien & 'Search Goods' eine '0' und 'Experience Goods' eine '1' zuweisen
germanX = germanX[(germanX.product_category == 'Music') | (germanX.product_category == 'Video DVD') | (germanX.product_category == 'Camera') | (germanX.product_category == 'PC')]
germanX['search_experience'] = np.where((germanX['product_category'] == 'Music') | (germanX['product_category'] == 'Video DVD'), 1, 0 )

#Helpfulness anhand Summe Helpful_Votes und Labelling
germanX['helpfulness_ratio'] = germanX['helpful_votes'] / germanX['total_votes']
germanX['helpfulness_label'] = np.where(germanX['helpfulness_ratio'] > 0.7, 1, 0)

#Berechnen des Sentiments mit TextBlob (Polarität & Subjektivität)
def subjectivity_func(review):
    try:
        return TextBlobDE(review).subjectivity
    except:
        return None

def polarity_func(review):
    try:
        return TextBlobDE(review).polarity
    except:
        return None

germanX['subjectivity'] = germanX['review_body'].apply(subjectivity_func)

germanX['polarity'] = germanX['review_body'].apply(polarity_func)

#Hinzufügen des Features Readability
def readability_func(review):
    try:
        return textstat.flesch_reading_ease(review)
    except:
        return None
    

germanX['readability'] = germanX['review_body'].apply(readability_func)

#Länge Headline & Review

germanX['len_headline'] = germanX['review_headline'].apply(len)

germanX['len_review'] = germanX['review_body'].apply(len)

#Verified Purchase in 1 bzw. 0 umwandeln
germanX['verified_purchase_binary'] = np.where(germanX['verified_purchase'] == 'Y', 1, 0)

del germanX['verified_purchase']


#Alter der Rezension in Tagen berechnen (Klassfikationsverfahren funktionieren leider nicht mit dem Age-Attribut)
"""
today = datetime.datetime(2020, 6, 20)

germanX['age'] = (today - pd.to_datetime(germanX['review_date'])).dt.days * 1


germanX.replace([np.inf, -np.inf], np.nan).dropna(how="any")
"""

#Entfernen weiterer nicht benötigter Attribute
del germanX['product_category']
del germanX['helpful_votes']
del germanX['total_votes']
del germanX['vine']
del germanX['review_headline']
del germanX['review_body']
del germanX['helpfulness_ratio']
del germanX['sentiment']
del germanX['review_date'] 

"""Klassifikation"""
#Train Test Split
X = germanX.drop('helpfulness_label', axis = 1).values
y = germanX['helpfulness_label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,test_size=0.2)

#Für Ergebnisse der einzelnen Klassifikationsverfahren, entsprechend Kommentar entfernen

"KNN"
classifier_KNN = KNeighborsClassifier(n_neighbors = 7)
classifier_KNN.fit(X_train, y_train)
predictions_KNN = classifier_KNN.predict(X_test)
print('KNN')
print('Accuracy score: '+str(accuracy_score(y_test,predictions_KNN)))
print('Precision score: '+str(precision_score(y_test,predictions_KNN)))
print(confusion_matrix(y_test, predictions_KNN))  


"""
"Naive Bayes"
classifier_GNB = GaussianNB()
classifier_GNB.fit(X_train, y_train)
predictions_GNB = classifier_GNB.predict(X_test)
print('GNB')
print('Accuracy score: '+str(accuracy_score(y_test,predictions_GNB)))
print('Precision score: '+str(precision_score(y_test,predictions_GNB)))
print(confusion_matrix(y_test, predictions_GNB))
"""

#SVM: Berechnung ist endlos, daher leider keine Ergebnisse.
"""
"Support Vector Machine"
classifier_SVM = svm.SVC()
classifier_SVM.fit(X_train, y_train)
predictions_SVM = classifier_SVM.predict(X_test)
print('SVM')
print('Accuracy score: '+str(accuracy_score(y_test,predictions_SVM)))
print('Precision score: '+str(precision_score(y_test,predictions_SVM)))
print(confusion_matrix(y_test, predictions_SVM))
"""

"""
"Decision Tree Classifier"
classifier_DTC = DecisionTreeClassifier()
classifier_DTC.fit(X_train, y_train)
predictions_DTC = classifier_DTC.predict(X_test)
print('DTC')
print('Accuracy score: '+str(accuracy_score(y_test,predictions_DTC)))
print('Precision score: '+str(precision_score(y_test,predictions_DTC)))
print(confusion_matrix(y_test, predictions_DTC))
"""

"""
"Linear Discriminant Analysis"
classifier_LDA = LinearDiscriminantAnalysis()
classifier_LDA.fit(X_train, y_train)
predictions_LDA = classifier_LDA.predict(X_test)
print('LDA')
print('Accuracy score: '+str(accuracy_score(y_test,predictions_LDA)))
print('Precision score: '+str(precision_score(y_test,predictions_LDA)))
print(confusion_matrix(y_test, predictions_LDA))
"""

"""
"Logistic Regression"
classifier_LR = LogisticRegression()
classifier_LR.fit(X_train, y_train)
predictions_LR = classifier_LR.predict(X_test)
print('LR')
print('Accuracy score: '+str(accuracy_score(y_test,predictions_LR)))
print('Precision score: '+str(precision_score(y_test,predictions_LR)))
print(confusion_matrix(y_test, predictions_LR))
"""
