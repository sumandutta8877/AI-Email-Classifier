from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import email
import re

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = stopwords.words('english')


"""Loading Data"""

data = pd.read_csv("/gdrive/My Drive/AI Email Classifier/emails.csv")

"""Extracting Catagories"""

catagories = []
for name in data['file']:
  name=name.split('/')
  catagories.append(name[1])

data['catagory'] = catagories
labels = list(data['catagory'].unique())


selected_catagories =  ['discussion_thread', 'personal', 'meetings', 'logistics', 'calendar', 'archiving', 'california', 'power', 'deal_communication', 'resumes']

data = data[data['catagory'].isin(selected_catagories)]

"""Extracting Message Body and Headers"""

#mail body
email_body = []

for mail in data['message']:
  mail = email.message_from_string(mail)  

  # getting message body  
  message_body = mail.get_payload()
  
  email_body.append(message_body)

data['message_body'] = email_body


#getting headers

headers = {"Date":[], "Subject":[], "X-Folder":[], "X-From":[], "X-To":[]}
for mail in data['message']:
  mail = email.message_from_string(mail)  

  #get other headers
  for header in headers.keys():
    headers[header].append(mail.get(header))

for key in headers.keys():
  data[key] = headers[key]
data['Date'] = pd.to_datetime(data['Date'])


"""Dropping empty rows"""

data.dropna(inplace = True)


"""Text cleaning"""

#Removing non alphanumeric characters
def clean_text(text):
  cleanText = text.lower()
  cleanText = re.sub(r'[\W\d]', " ", cleanText)  

  return cleanText

#Removing stop words
def stop_word_removal(text):
  tokens = [token for token in text if token not in stopwords]
  return tokens

#Tokenizing
def tokenize(text):
  tokens = text.split(" ")
  return tokens

headers = ["Subject", "X-Folder", "X-From", "X-To", "message_body"]

tokens = [] 

for i in range(data.shape[0]):
  tokens_i = []
  for header in headers:
    tokens_h = clean_text(data[header].values[i])
    tokens_h = tokenize(tokens_h)
    tokens_h = stop_word_removal(tokens_h)

    tokens_i.extend(tokens_h)

  tokens.append(tokens_i)

for i in range(len(tokens)):
  tokens[i].remove("")
  tokens[i] = " ".join(tokens[i])

data['final_text'] = tokens


"""Applying Train-Test split"""

x_train, x_test, y_train, y_test = model_selection.train_test_split(data['final_text'], data['catagory'], test_size=0.3)

"""Encoding data catagories"""

Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

"""Applying TF-IDF vectorizer on final text"""

Tfidf_vect = TfidfVectorizer()

Tfidf_vect.fit(data['final_text'])

x_train = Tfidf_vect.transform(x_train)
x_test = Tfidf_vect.transform(x_test)


"""Applying SVM"""

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(x_train, y_train)

predictions_SVM = SVM.predict(x_test)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)