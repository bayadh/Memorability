import os
import pandas as pd
import keras
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
# spearman score
def Get_score(Y_pred,Y_true):
    
    Y_pred = np.squeeze(Y_pred)
    Y_true = np.squeeze(Y_true)
    if Y_pred.shape != Y_true.shape:
        print('shapes do not match!')
    else:
        if len(Y_pred.shape) == 1:
            Res = pd.DataFrame({'Y_true':Y_true,'Y_pred':Y_pred})
            score_mat = Res[['Y_true','Y_pred']].corr(method='spearman',min_periods=1)
            print(' Spearman correlation coefficient is: %.3f' % score_mat.iloc[1][0])
        else:
            for ii in range(Y_pred.shape[1]):
                Get_score(Y_pred[:,ii],Y_true[:,ii])


def spearman_corr(x_pred,x_true):
    a = pd.DataFrame()
    a['true'] = x_true
    a['pred'] = x_pred
    res = a[['true','pred']].corr(method='spearman',min_periods=1)
    return res.iloc[0,1]



# load labels and captions
def read_caps(fname):
    vn = []
    cap = []
    df = pd.DataFrame();
    with open(fname) as f:
        for line in f:
            pairs = line.split()
            vn.append(pairs[0])
            cap.append(pairs[1])
        df['video']=vn
        df['caption']=cap
    return df

# load the captions
cap_path = '../dev-set_video-captions.txt'
df_captions=read_caps(cap_path)

# load the ground truth values
label_path = '../ground-truth/'
labels=pd.read_csv(label_path+'ground-truth_dev-set.csv')

df = df_captions.merge(labels,on=["video"],how="inner")


cVect=CountVectorizer()
X_CVect=cVect.fit_transform(df['caption'])
X_arrseq=X_CVect.toarray()
print(X_arrseq[0])

Y=df[['short-term_memorability','long-term_memorability']].values  #targets
X=X_arrseq #input
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42) #split data
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('Y_train', Y_train.shape)
print('Y_test', Y_test.shape)
n_cols = X_train.shape[1] #nbr of columns in predictors

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
# Add the second layer
model.add(Dense(100, activation='relu'))

# Add the output layer
model.add(Dense(2))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])



# Fit the model
r=model.fit(X_train, Y_train, validation_split=0.3, epochs=40)

# Compile the model
model.compile(optimizer='adam', loss = 'mean_squared_error')

# Print the loss
print("Loss function: " + model.loss)

# Fit the model
r = model.fit(X_train,Y_train,epochs=40,validation_data=(X_test,Y_test))




predictions = model.predict(X_test)
print(predictions)
Get_score(predictions, Y_test) #  Spearman scores
print('the short-term score is {:.3f}'.format(spearman_corr(Y_test[:,0],predictions[:,0])))
print('the long-term score is {:.3f}'.format(spearman_corr(Y_test[:,1],predictions[:,1])))
