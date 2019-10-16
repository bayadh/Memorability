import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
import os


from tensorflow import set_random_seed



def read_array(fname):
    with open(fname) as f:
        for line in f:
            pairs = line.split(',')
    #print(pairs)
        return [float(item) for item in pairs]

def spearman_corr(x_pred,x_true):
    a = pd.DataFrame()
    a['true'] = x_true
    a['pred'] = x_pred
    res = a[['true','pred']].corr(method='spearman',min_periods=1)
    return res.iloc[0,1]

#Load aesthetic features
path_AF_Mean = '../features/aesthetic_visual_features/aesthetic_feat_dev-set_mean'
path_AF_Median = '../features/aesthetic_visual_features/aesthetic_feat_dev-set_median'

# Load video related features
vn_mean = os.listdir(path_AF_Median)

# stack the video names in dataframe
df = pd.DataFrame()
df['video'] = [os.path.splitext(vn)[0]+'.webm' for vn in vn_mean]

# read the aesthetic feat (mean and media) in dataframe
df['AF_mean'] = [ read_array(path_AF_Mean+'/'+vn[:-5]+'.txt') for vn in df['video']]
df['AF_median'] = [ read_array(path_AF_Median+'/'+vn[:-5]+'.txt') for vn in df['video']]

# load the ground truth values
label_path = '../ground-truth/'
labels=pd.read_csv(label_path+'ground-truth_dev-set.csv')

# merge
df_final = pd.merge(df,labels,on='video')

# use the AF_median 
X =np.array([col for col in df_final['AF_median'].values ])
Y = df_final[['short-term_memorability','long-term_memorability']].values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=124)



# model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(109,)))
# Add the second layer
model.add(Dense(100, activation='relu'))

# Add the output layer
model.add(Dense(2))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
history = model.fit(X_train,Y_train,epochs=20,validation_split=0.2)
Y_pred_test=model.predict(X_test)


print('the short-term score is {:.3f}'.format(spearman_corr(Y_test[:,0],Y_pred_test[:,0])))
print('the long-term score is {:.3f}'.format(spearman_corr(Y_test[:,1],Y_pred_test[:,1])))




