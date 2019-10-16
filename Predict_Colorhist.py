import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import os


from tensorflow import set_random_seed




def read_ColorHistogram(fname):
    RGB_Hist = np.zeros((3,256))
    with open(fname) as f:
        i_l = 0 # line index
        for line in f:
            pairs = line.split()
            hist_dict = {int(p.split(':')[0]):float(p.split(':')[1]) for p in pairs}
            for idx in hist_dict.keys():
                RGB_Hist[i_l,idx] = hist_dict[idx]
            i_l += 1
    return RGB_Hist

def spearman_corr(x_pred,x_true):
    a = pd.DataFrame()
    a['true'] = x_true
    a['pred'] = x_pred
    res = a[['true','pred']].corr(method='spearman',min_periods=1)
    return res.iloc[0,1]

#Load aesthetic features
path_hist = '../features/ColorHistogram/'

# Load video related features
vn_hist = os.listdir(path_hist)

# stack the video names in dataframe
df = pd.DataFrame()
df['video'] = [os.path.splitext(vn)[0]+'.webm' for vn in vn_hist]
df['v'] = [os.path.splitext(vn)[0] for vn in vn_hist]

# read the aesthetic feat (mean and media) in dataframe
df['ColorHist'] = [ read_ColorHistogram(path_hist+vn+'.txt') for vn in df['v']]
print(df['ColorHist'])

# load the ground truth values
label_path = '../ground-truth/'
labels=pd.read_csv(label_path+'ground-truth_dev-set.csv')



X = np.stack(df['ColorHist'].values)

print('********')
print(X)
Y = labels[['short-term_memorability','long-term_memorability']].values
print('********')
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=124)
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('Y_train', Y_train.shape)
print('Y_test', Y_test.shape)
n_cols = X_train.shape[1] #nbr of columns in predictors


# model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
# Add the second layer
model.add(Dense(100, activation='relu'))

# Add the output layer
model.add(Dense(2))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X_train,Y_train,epochs=40,validation_split=0.2)
Y_pred_test=model.predict(X_test)


print('the short-term score is {:.3f}'.format(spearman_corr(Y_test[:,0],Y_pred_test[:,0])))
print('the long-term score is {:.3f}'.format(spearman_corr(Y_test[:,1],Y_pred_test[:,1])))





