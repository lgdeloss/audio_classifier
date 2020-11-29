import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 
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

# Print shapes
print("X.shape: " + str(X.shape))
print("y.shape: " + str(y.shape))

# Encode the classification labels
le = preprocessing.LabelEncoder()
yy = le.fit_transform(y)

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.2, random_state = 42)


# num_rows = 40
# num_columns = 174
# num_channels = 1
# x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
# x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)

num_labels = 2
filter_size = 2

# Construct model 
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, input_shape=(x_train.shape[0],2868, 13), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

# model.add(Conv2D(filters=32, kernel_size=2, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(Conv2D(filters=64, kernel_size=2, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))

# model.add(Conv2D(filters=128, kernel_size=2, activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Display model architecture summary 
model.summary()

# Calculate pre-training accuracy 
score = model.evaluate(x_test, y_test, verbose=1)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

from keras.callbacks import ModelCheckpoint 
from datetime import datetime 

num_epochs = 72
num_batch_size = 256

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.basic_cnn.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)


# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: ", score[1])

score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: ", score[1])