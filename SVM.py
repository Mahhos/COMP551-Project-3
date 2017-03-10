%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
import pandas as pd
import csv
from sklearn.pipeline import Pipeline
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

original_train_x = np.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
original_train_y = np.load('tinyY.npy') 
testX = np.load('tinyX_test.npy') # (6600, 3, 64, 64)

idx = random.sample(range(np.size(original_train_y)), np.size(original_train_y))

trainX = original_train_x[idx]
trainY = original_train_y[idx] 

def to_gray(color_img):
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    return gray
img_gray = to_gray(trainX[60].transpose(2,1,0))
plt.imshow(img_gray, cmap='gray');


# -------------------------------------------------------------------------------- #
# ----------- Changeing training data to gray scale -------------------------------------------- #
# -------------------------------------------------------------------------------- #


num = trainX.size
graytrainx = []
for i in range(0, 26344):
    print("changing to Gray" + " " + str(i))
    graytrainx.append(to_gray(trainX[i].transpose(2,1,0)))
graytrainxArr = np.asarray(graytrainx)
twoDtrainxArr = graytrainxArr.transpose(2,0,1).reshape(26344 ,-1)

# -------------------------------------------------------------------------------- #
# ----------- Changeing test data to gray scale -------------------------------------------- #
# -------------------------------------------------------------------------------- #
graytestx = []
for i in range(0, 6600):
    print("changing to Gray" + " " + str(i))
    graytestx.append(to_gray(testX[i].transpose(2,1,0)))
graytestxArr = np.asarray(graytestx)
twoDtestxArr = graytestxArr.transpose(2,0,1).reshape(6600 ,-1)




# Instantiate and train a support vector machine
svm = svm.SVC(kernel = 'rbf')

# Train the SVM
x_train, x_val, y_train, y_val = train_test_split(twoDtrainxArr , trainY, test_size = 0.8)


print("Training SVM...")
svm.fit(x_train, y_train)
print("Trained SVM.")

# Predict on training data
print("Predicting training set...")
pred_train = svm.predict(x_train)
pred_train_df = pd.DataFrame({
	'prediction': pred_train,
	'class': y_train
	})
pred_train_df.to_csv("trainprediction.csv",
	index_label = "id", header = ["prediction", "class"])

# Predict on validation data
print("Predicting validation set...")
pred_val = svm.predict(x_val)
pred_val_df = pd.DataFrame({
	'prediction': pred_val,
	'class': y_val
	})
pred_val_df.to_csv("valpredictions.csv",
	index_label = "id", header = ["prediction", "class"])

# Predict on test data
print("Predicting test set...")

pred_test = svm.predict(twoDtestxArr)
pred_test_df = pd.DataFrame(pred_test)
pred_test_df.to_csv("predictions.csv",
	index_label = "id")

