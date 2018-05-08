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
from tensorflow.contrib import learn
from CNN import *


print("reading the dataset.....")
data = pd.read_csv('../data/datasets/combined_dataset.csv', encoding = "utf-8")
data = data[['class','tweet']]
data = data.reindex(np.random.permutation(data.index))
data = data[1:1000]

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


all_tweets_cleaned = list(pos_neg_data["cleaned_tweet_text_only"])

sentences_lengths = [len(x.split()) for x in all_tweets_cleaned]
max_document_len = max(sentences_lengths)
see("max_document_len",max_document_len)

vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_len)
vocab_processor.fit(all_tweets_cleaned)
vocab_size = len(vocab_processor.vocabulary_)
see("datatset vocab_size", vocab_size)

#vocab_dict = vocab_processor.vocabulary_._mapping
#np.array([all_tweets_cleaned[1]])
#test = np.transpose(np.array([all_tweets_cleaned[1]]))
#print(test.shape)
#see("all_tweets_cleaned[1]", test)
#see("sent", list(vocab_processor.transform(test)))


embedding_size = 300
learning_rate = 0.1
discrete_featuers_size = len(pos_neg_data.iloc[0]["handed_features"])
print(discrete_featuers_size)

cnn_model = TextCNN(embedding_size=embedding_size, filter_sizes=[2, 3, 4], num_classes=2, num_filters=4,
                    vocab_size=vocab_size, sequence_length=max_document_len,
                    discrete_features_size=discrete_featuers_size, l2_reg_lambda=0.1)


train, test = train_test_split(pos_neg_data, test_size=0.1)
print("training set size = " + str(len(train)))
print("testing set size = " + str(len(test)))


def generate_batch_data(train_data, batch_size=10):
    batch_data = train.sample(n=batch_size, replace=False)
    x_train = np.transpose(batch_data["cleaned_tweet_text_only"])
    x_train_features = np.array(list(batch_data["handed_features"]))
    y_train = np.array(list(batch_data["label_encoded"]))
    xi_train = np.array(list(vocab_processor.transform(x_train)))
    #print(xi_train.shape)
    #print(x_train_features.shape)
    #print(y_train.shape)
    return xi_train,x_train_features, y_train


def train_step(batch_id, x_batch, y_batch, x_feat, dropout_keep_prob=0.8):
    feed_dict = {cnn_model.input_x: x_batch,
                 cnn_model.input_y: y_batch,
                 cnn_model.discrete_features: x_feat,
                 cnn_model.dropout_keep_prob: dropout_keep_prob}
    loss, accuracy = sess.run([cnn_model.loss, cnn_model.accuracy], feed_dict)
    print("bacht_no {:g}, loss {:g}, acc {:g}".format(batch_id, loss, accuracy))
    # print(cnn_model.combined_features.shape)
    return loss, accuracy


def dev_step(x_batch, y_batch, x_feat):
    return train_step(1, x_batch, y_batch, x_feat, 1)



tf.train.AdagradOptimizer(learning_rate).minimize(cnn_model.loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

iterations = 1000
batch_size = 100
evaluate_every = 100

train_loss = []
train_acc = []
test_loss = []
test_acc = []
i_data = []
ii_data = range(iterations)

for batch_id in ii_data:
    train_shuffled = train.reindex(np.random.permutation(train.index))
    x_batch, x_feat, y_batch = generate_batch_data(train_shuffled, batch_size)
    loss, accuracy = train_step(batch_id, x_batch, y_batch, x_feat)
    train_loss.append(loss)
    train_acc.append(accuracy)

    if (batch_id % evaluate_every == 0):
        print("\n------------- Evaluation -------------------")
        test_shuffled = test.reindex(np.random.permutation(test.index))
        x_dev, x_dev_feat, y_dev = generate_batch_data(test_shuffled, test.shape[0])
        loss, accuracy = dev_step(x_dev, y_dev, x_dev_feat)
        test_loss.append(loss)
        test_acc.append(accuracy)
        i_data.append(batch_id + 1)
        print("----------------------------------------------")


# Plot loss over time
plt.plot(ii_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(ii_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()