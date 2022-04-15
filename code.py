import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns 
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
dataset =pd.read_csv(r"C:\Users\abdoe\.spyder-py3\spam_ham_dataset.csv")
dataset.head(10)
dataset.tail(10)
dataset.describe()

spam = dataset[dataset['label_num'] == 1]
ham = dataset[dataset['label_num']==0]

sns.countplot(dataset['label_num'])


stopwords_set = set(stopwords.words('english'))
stopwords_set.add('subject')
words = []
for i in range(0, len(dataset)):
    text = re.sub('[^a-zA-Z]', ' ', dataset['text'][i])
    text = text.lower()
    text = text.split()
    ps = PorterStemmer()
    text = [ps.stem(word) for word in text if not word in stopwords_set]
    text = ' '.join(text)
    words.append(text)
    
    

sns.countplot(dataset['label_num'])
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
spam_ham = vectorizer.fit_transform(words)
print(spam_ham.toarray())



spam_ham.shape
X= spam_ham
Y= dataset['label_num'].values


print(Y)

from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test =train_test_split(X, Y, test_size = 0.2, random_state = 42)


from sklearn.naive_bayes import MultinomialNB
NB_classifier =MultinomialNB()
NB_classifier.fit(X_train , Y_train)
Y_pred_train = NB_classifier.predict(X_train)
print(Y_pred_train)
Y_pred_test = NB_classifier.predict(X_test)


print(Y_pred_test)
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
cm_test = confusion_matrix(Y_test, Y_pred_test)
cm_train = confusion_matrix(Y_train , Y_pred_train)

print(cm_test)
print(cm_train)
sns.heatmap(cm_train , annot = True)
sns.heatmap(cm_test, annot = True)
accuracy_score(Y_test , Y_pred_test)*100
print(classification_report(Y_pred_test, Y_test))