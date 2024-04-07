from django.db import models
from django.db import models

# Create your models here.
from django.db import models

# Create your models here.
from django.db import models


import numpy as np
import pickle

import json
from PIL import Image


# Testing phase
dt = pickle.load(open(r"C:\Users\ragzv\Music\FRONT END\DT_tfidf.pkl", 'rb'))
rf= pickle.load(open(r"C:\Users\ragzv\Music\FRONT END\RF_tfidf.pkl", 'rb'))


tfidf_feature = pickle.load(open(r"C:\Users\ragzv\Music\FRONT END\new_tfidf.pkl", 'rb'))


def predict(text,algo): 
	text = [text]
	filter_text = tfidf_feature.transform(text)
	print(filter_text.shape)
	if algo=='dt':
		y_pred=dt.predict(filter_text)
		return y_pred[0]
	else:
		y_pred=rf.predict(filter_text)
		return y_pred[0]

