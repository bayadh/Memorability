import pandas as pd
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import os
import numpy as np


def Get_score(Y_pred,Y_true):
    Y_pred = np.squeeze(Y_pred)
    Y_true = np.squeeze(Y_true)
    if Y_pred.shape != Y_true.shape:
        print('Input shapes do not match!')
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





#read HMP
def read_HMP(fname):
    with open(fname) as f:
        for line in f:
            pairs=line.split()
            HMP_temp = { int(p.split(':')[0]) : float(p.split(':')[1]) for p in pairs}
    # there are 6075 bins, fill zeros
    HMP = np.zeros(6075)
    for idx in HMP_temp.keys():
        HMP[idx-1] = HMP_temp[idx]
    return HMP


# load the ground truth values
label_path = '../ground-truth/'
labels=pd.read_csv(label_path+'ground-truth_dev-set.csv')

vid = labels.video.values

#load HMP 
dir_hmp = '../features/HMP/'
Features = pd.DataFrame({'video': vid,
                   'HMP': [read_HMP(dir_hmp+os.path.splitext(item)[0]+'.txt') for item in vid],
                       })


X = np.stack(Features['HMP'].values)
Y = labels[['short-term_memorability','long-term_memorability']].values

X_train, X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('Y_train', Y_train.shape)
print('Y_test', Y_test.shape)



# model
model = Sequential()
model.add(Dense(50, activation='relu',input_shape=(X_train.shape[1],)))

# Add the second layer
model.add(Dense(100, activation='relu'))

# Add the output layer
model.add(Dense(2))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
   
# Fit the model
history=model.fit(X_train, Y_train, validation_split=0.3, epochs=40)

# predictions
predictions= model.predict(X_test)
print(predictions)
Get_score(predictions, Y_test)
print('the short-term score is {:.3f}'.format(spearman_corr(Y_test[:,0],predictions[:,0])))
print('the long-term score is {:.3f}'.format(spearman_corr(Y_test[:,1],predictions[:,1])))
