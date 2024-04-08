#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#list of useful imports that  I will use
get_ipython().run_line_magic('matplotlib', 'inline')
import os

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

import seaborn as sns
import random
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

from sklearn.impute import SimpleImputer
from sklearn_pandas import DataFrameMapper



import warnings
warnings.filterwarnings('ignore')


# In[2]:


data1 = pd.read_csv(r'C:\Users\ragzv\Music\Personality_pred\dataset.csv')


# In[3]:


data1


# In[4]:


data = data1[['type', 'posts', 'Disorder']]


# In[5]:


data.head()


# In[6]:


data.shape


# In[7]:


#Check the data
data.info()


# In[8]:


#Check the missing values in the data
data.isnull().sum()


# In[9]:


data.dropna(inplace=True)


# In[10]:


data.isnull().sum()


# In[11]:


data['Disorder'].value_counts()


# In[12]:


data['Disorder']=data['Disorder'].replace({'bipolar_disorder, depression, ptsd, seasonal_affective_disorder':'bipolar_disorder','suicide_(attempt), anxiety, depression, eating, panic, schizophrenia, bipolar_disorder, ptsd':'adhd,suicide_(attempt),anxiety,stress','suicide_(ideation), depression, anxiety, stress':'adhd,suicide_(attempt),anxiety,stress','suicide_(ideation), stress':'adhd,suicide_(attempt),anxiety,stress','suicide_(attempt), suicide_(ideation), depression':'adhd,suicide_(attempt),anxiety,stress','stress, stress_(stressor_and_subjects) ':'suicide_(attempt),anxiety,stress','adhd, anxiety, autism, bipolar_disorder':'adhd,suicide_(attempt),anxiety,stress','eating, eating_(recovery)':'eating, depression','depression, eating':'eating, depression','anxiety':'suicide_(attempt),anxiety,stress','anxiety, borderline_personality_disorder, bipolar_disorder, opiate_addiction, self_harm, aspergers, autism, alcoholism, opiate_usage, schizophrenia, suicide_(ideation)':'bipolar_disorder,anxiety','depression, trauma, bipolar_disorder, ptsd, psychosis, eating, self_harm, rape_(survivors), panic, anxiety_(social), suicide_(ideation)':'bipolar_disorder,anxiety','stress':'suicide_(attempt),anxiety,stress','postpartum_depression':'eating, depression','depression_(symptoms)':'eating, depression','borderline_personalitiy_disoreder':'borderline_personality_disorder','suicide_(ideation)':'suicide_(attempt),anxiety,stress','eating':'eating, depression','aggression':'cyberbullying,aggression','cyberbullying':'cyberbullying,aggression','adhd':'adhd,suicide_(attempt),anxiety,stress','adhd, anxiety, bipolar_disorder, depression, eating, ocd, ptsd, schizophrenia, seasonal_affective_disorder':'ptsd','ptsd':'adhd,suicide_(attempt),anxiety,stress','adhd,suicide_(attempt),anxiety,stress':'adhd,suicide_(attempt),anxiety,stress,ptsd','self_harm':'self_harm,self_esteem','self_esteem':'self_harm,self_esteem','bipolar_disorder':'bipolar_disorder,anxiety','eating_(recovery)':'eating, depression','depression, substance_use, sleep_disorder, eating':'eating, depression','stress, stress_(stressor_and_subjects)':'suicide_(attempt),anxiety,stress','suicide_(ideation), imminent_death, depression, loneliness':'suicide_(attempt),anxiety,stress','life_satisfaction, depression':'depression','sentiment':'self_harm,self_esteem','cognitive_distortion':'mental_health_(combined),cognitive_distortion','mental_health_(combined)':'mental_health_(combined),cognitive_distortion','antisocial_behavior':'self_harm,self_esteem','self_harm,self_esteem':'self_harm,self_esteem,antisocial_behavior'})
data['Disorder'].value_counts()


# In[13]:


from sklearn.utils import resample
# Separate majority and minority classes
df1 = data[data['Disorder']== 'borderline_personality_disorder']
df2 = data[data['Disorder']== 'suicide_(attempt),anxiety,stress']

# Downsample majority class and upsample the minority class
df1_upsampled = resample(df1, replace=True,n_samples=800,random_state=123) 
df2_downsampled = resample(df2, replace=True,n_samples=800,random_state=123)

# Combine minority class with downsampled majority class
data1 = pd.concat([df1_upsampled, df2_downsampled])

# Display new class counts
data1['Disorder'].value_counts()


# In[14]:


data1.head(5)


# In[15]:


data1['type'].value_counts()


# In[16]:


from sklearn.utils import resample
# Separate majority and minority classes
df1 = data1[data1['type']== 'INFP']
df2 = data1[data1['type']== 'INFJ']
df3 = data1[data1['type']== 'INTP']
df4 = data1[data1['type']== 'INTJ']
df5 = data1[data1['type']== 'ENTP']

 
# Downsample majority class and upsample the minority class
df1_upsampled = resample(df1, replace=True,n_samples=150,random_state=123) 
df2_downsampled = resample(df2, replace=True,n_samples=150,random_state=123)
df3_upsampled = resample(df3, replace=True,n_samples=150,random_state=123) 
df4_downsampled = resample(df4, replace=True,n_samples=150,random_state=123)
df5_upsampled = resample(df5, replace=True,n_samples=150,random_state=123) 

# Combine minority class with downsampled majority class
df_upsampled = pd.concat([df1_upsampled, df2_downsampled,df3_upsampled, df4_downsampled,df5_upsampled])
# Display new class counts
df_upsampled['type'].value_counts()


# In[17]:


# shuffle the DataFrame rows 
data2= df_upsampled.sample(frac = 1)


# In[18]:


import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# In[19]:


stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"]


# In[20]:


data2['posts'].head(5)


# In[21]:


print("printing some random comments")
print(7, data2['posts'].values[7])
print(234, data2['posts'].values[234])
print(17, data2['posts'].values[17])


# In[22]:


# Combining all the above stundents 
from tqdm import tqdm
def preprocess_text(text_data):
    preprocessed_text = []
    # tqdm is for printing the status bar
    for sentance in tqdm(text_data):
        sent = decontracted(sentance)
        sent = sent.replace('\\r', ' ')
        sent = sent.replace('\\n', ' ')
        sent = sent.replace('\\"', ' ')
        sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
       
        sent = ' '.join(e for e in sent.split() if e.lower() not in stopwords)
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text


# In[23]:


preprocessed_text = preprocess_text(data2['posts'].values)


# In[24]:


print("printing some random comments")
print(7, preprocessed_text[7])
print(234, preprocessed_text[234])
print(17, preprocessed_text[17])


# In[25]:


from sklearn.preprocessing import LabelEncoder


# In[26]:


data2['type'].value_counts()


# In[27]:


data2.head(10)


# In[28]:


data2.tail(10)


# In[29]:


data2.shape


# In[30]:


x = data2[['posts']]


# In[31]:


le = LabelEncoder()
y = le.fit_transform(data2['type'])
y1= le.fit_transform(data2['Disorder'])
y1 = np.array(y1)
y = np.array(y)


# In[32]:


from sklearn.model_selection import train_test_split
#Breaking into Train and test
X_train, X_test, y_train, y_test = train_test_split(preprocessed_text, y, test_size=0.3,stratify=y ,random_state=42)


# In[33]:


X_train


# In[34]:


X_test


# In[35]:


pd.DataFrame(X_test).to_csv(r"C:\Users\ragzv\Music\test.csv",index=False)


# In[36]:


y_train.shape


# In[37]:


y_test


# ## Featuraization:- TF-IDF

# In[38]:


import pickle
from sklearn import preprocessing

tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=10)
tfidf.fit(X_train)
import pickle
filename = r'C:\Users\ragzv\Music\FRONT END\new_tfidf.pkl'
pickle.dump(tfidf, open(filename, 'wb'))# fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_tfidf =tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#Normalize Data
X_train_tfidf = preprocessing.normalize(X_train_tfidf)
print("Train Data Size: ",X_train_tfidf.shape)

#Normalize Data
X_test_tfidf = preprocessing.normalize(X_test_tfidf)
print("Test Data Size: ",X_test_tfidf.shape)


# ### Random Forest with TF-IDF

# In[39]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve

dept = [1, 5, 10, 50, 100, 500, 1000]
n_estimators =  [20, 40, 60, 80, 100, 120]

param_grid={'n_estimators':n_estimators , 'max_depth':dept}
clf = RandomForestClassifier()
model = GridSearchCV(clf,param_grid,scoring='accuracy',n_jobs=-1,cv=3)
model.fit(X_train_tfidf,y_train)
print("optimal n_estimators",model.best_estimator_.n_estimators)
print("optimal max_depth",model.best_estimator_.max_depth)
optimal_max_depth = model.best_estimator_.max_depth
optimal_n_estimators = model.best_estimator_.n_estimators


# In[40]:


from sklearn.metrics import accuracy_score
#training our model for max_depth=100,n_estimators = 120
clf = RandomForestClassifier(max_depth = optimal_max_depth,n_estimators = optimal_n_estimators)
clf.fit(X_train_tfidf,y_train)

import pickle
filename = r'C:\Users\ragzv\Music\FRONT END\RF_tfidf.pkl'
pickle.dump(clf, open(filename, 'wb'))


pred_test =clf.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, pred_test)
pred_train = clf.predict(X_train_tfidf)
train_accuracy =accuracy_score(y_train,pred_train)



print("AUC on Test data is " +str(accuracy_score(y_test,pred_test)))
print("AUC on Train data is " +str(accuracy_score(y_train,pred_train)))

print("---------------------------")

# Code for drawing seaborn heatmaps
class_names = ['INFP personality No borderline_personality_disorder','INFJ with borderline_personality_disorder ','INTP No borderline_personality_disorder','INTJ No borderline_personality_disorder','ENTP With borderline_personality_disorder']
df_heatmap = pd.DataFrame(confusion_matrix(y_test, pred_test.round()), index=class_names, columns=class_names )
fig = plt.figure( )
heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d")


# In[41]:


all_model_result = pd.DataFrame(columns=['METHOD', 'Classifier' , 'Train-Accuracy', 'Test-Accuracy' ])
new = ['TFIDF ','Random forest-Classifier',train_accuracy, test_accuracy]
all_model_result.loc[0] = new


# In[42]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV

dept = [1, 5, 10, 50, 100, 500,800, 1000]
min_samples =  [5, 10, 100, 500]


param_grid={'min_samples_split':min_samples , 'max_depth':dept}
clf = DecisionTreeClassifier()
model = GridSearchCV(clf,param_grid,scoring='accuracy',n_jobs=-1,cv=3)
model.fit(X_train_tfidf,y_train)
print("optimal min_samples_split",model.best_estimator_.min_samples_split)
print("optimal max_depth",model.best_estimator_.max_depth)
optimal_max_depth = model.best_estimator_.max_depth
optimal_min_samples_split = model.best_estimator_.min_samples_split


# In[43]:


#Testing AUC on Test data
dt = DecisionTreeClassifier(max_depth =500,min_samples_split =5)

dt.fit(X_train_tfidf,y_train)

import pickle
filename = r'C:\Users\ragzv\Music\FRONT END\DT_tfidf.pkl'
pickle.dump(dt, open(filename, 'wb'))

#predict on test data and train data
 
y_predtestd = dt.predict(X_test_tfidf)
y_predtraind = dt.predict(X_train_tfidf)

pred_test =dt.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, pred_test)
pred_train = dt.predict(X_train_tfidf)
train_accuracy =accuracy_score(y_train,pred_train)


print('*'*35)

#accuracy on training and testing data

print('the accuracy on testing data',accuracy_score(y_test,y_predtestd))
print('the accuracy on training data',accuracy_score(y_train,y_predtraind))
train0 = accuracy_score(y_train,y_predtraind)
test0 = accuracy_score(y_test,y_predtestd)

print('*'*35)
# Code for drawing seaborn heatmaps
class_names = ['INFP personality No borderline_personality_disorder','INFJ with borderline_personality_disorder ','INTP No borderline_personality_disorder','INTJ No borderline_personality_disorder','ENTP With borderline_personality_disorder']
df_heatmap = pd.DataFrame(confusion_matrix(y_test, pred_test.round()), index=class_names, columns=class_names )
fig = plt.figure( )
heatmap = sns.heatmap(df_heatmap, annot=True, fmt="d")


# In[44]:


new = ['TFIDF ','DECISION TREE',train0, test0]
all_model_result.loc[1] = new


# In[45]:


all_model_result


# In[ ]:




