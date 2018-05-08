from preprocessor import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.feature_extraction.text import  CountVectorizer
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.stem.arlstem import ARLSTem
from nltk.classify import SklearnClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB

print("reading the dataset.....")
data = pd.read_csv('../data/datasets/combined_dataset.csv', encoding = "utf-8")
data = data[['class','tweet']]
data = data.reindex(np.random.permutation(data.index))
data = data[1:6000]

print("cleaning the dataset....")
data['cleaned_tweet'] = data['tweet'].map(lambda v: clean_sentence(v)[0])
data['cleaned_tweet_text_only'] = data['tweet'].map(lambda v: clean_sentence(v)[1])

print("building features....")
all_document = data['cleaned_tweet']
tfidf_vectorizer = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=word_tokenize)
tfidf_model = tfidf_vectorizer.fit(all_document)
data['handed_features'] = data['cleaned_tweet'].map(lambda v: featurize(v, tfidf_model))
pos_neg_data = data[data['class'] != 'neutral']
pos_neg_data['label_encoded'] = pos_neg_data['class'].map(encode_class_labels)
print("finish processing...")

train, test = train_test_split(pos_neg_data, test_size=0.2)
x_train = list(train["handed_features"])
y_train = list(train["class"])
print("training set size = "+ str(len(x_train)))
x_test = list(test["handed_features"])
y_test = list(test["class"])
print("testing set size = "+ str(len(x_test)))


nb = MultinomialNB()
nb.fit(x_train, y_train)
preds = nb.predict(x_test)

print("\naccuracy = ", calculate_accuracy(preds, y_test))
print('\nconfusion_matrix')
print(confusion_matrix(y_test, preds))
print('\nclassification_report')
print(classification_report(y_test, preds))