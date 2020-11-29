import os
import numpy as np
import pandas as pd
import scipy.io.wavfile
import python_speech_features as psf
from sklearn.metrics import accuracy_score
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import scikitplot as skplt

def get_file_features(wav_fname, num_ceps=13):
    """
    Extract mfcc features from a file.
    """
    # read wave
    fs, sig = scipy.io.wavfile.read(wav_fname)

    # get mfccs
    mfccs = psf.mfcc(sig, samplerate=fs, winlen=0.025, winstep=0.01,
                     numcep=num_ceps, nfilt=26, nfft=512, lowfreq=0,
                     highfreq=None, preemph=0.97, ceplifter=22,
                     appendEnergy=False)

    # compute mfcc means
    mfcc_means = np.round(mfccs.mean(axis=0), 3)
    return mfcc_means 

# features
features=[]

human_pathlist = Path('./human').rglob('*.wav')
for path in human_pathlist:
     class_label='human'
     # because path is object not string
     path_in_str = str(path)
     features.append([get_file_features(path_in_str), class_label])


computer_pathlist = Path('./computer').rglob('*.wav')
for path in computer_pathlist:
     class_label='computer'
     # because path is object not string
     path_in_str = str(path)
     features.append([get_file_features(path_in_str), class_label])

# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

print('Finished feature extraction from ', len(featuresdf), ' files')

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())

# Encode the classification labels
le = preprocessing.LabelEncoder()
yy = le.fit_transform(y)
# yy= yy[1]
# split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)

# Print shapes
print("X.shape: " + str(X.shape))
print("y.shape: " + str(y.shape))


# Fit a Random Forest model 
rf = RandomForestClassifier(max_depth=2, random_state=0)
rf_probas=rf.fit(x_train, y_train).predict_proba(x_test)
# y_pred = pd.Series(model.predict(x_test))
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred)*100)
# print("Precision:", metrics.precision_score(y_test, y_pred, average="binary", pos_label=0)*100)
# print("Recall:", metrics.recall_score(y_test, y_pred, average="binary", pos_label=0)*100)


## Fit a Logistical Regression model
lr = LogisticRegression()
lr_probas=lr.fit(x_train, y_train).predict_proba(x_test)
# y_pred = pd.Series(model.predict(x_test))
# print("Accuracy:", metrics.accuracy_score(y_test, y_pred)*100)
# print("Precision:", metrics.precision_score(y_test, y_pred, average="binary", pos_label=0)*100)
# print("Recall:", metrics.recall_score(y_test, y_pred, average="binary", pos_label=0)*100)

# Fit an SVM model
svm = SVC(kernel = 'sigmoid', probability=True)
svm_scores=svm.fit(x_train, y_train).decision_function(x_test)
# print(accuracy_score(model.predict(X_val), y_val)*100)

# Fit a Gaussian Naive Bayes model
nb = GaussianNB()
nb_probas = nb.fit(x_train, y_train).predict_proba(x_test)

# Plot
# probas = model.predict_proba(x_test)
# skplt.metrics.plot_roc(y_test, probas)
# skplt.metrics.plot_precision_recall_curve(y_test, probas)
# skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True)
# plt.show()


probas_list = [rf_probas, lr_probas, nb_probas,svm_scores]
clf_names = ['Random Forest', 'Logistic Regression', 'Gaussian Naive Bayes','Support Vector Machine']
skplt.metrics.plot_calibration_curve(y_test, probas_list,clf_names)
plt.show()