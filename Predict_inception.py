import pandas as pd
import numpy as np
import os
import keras
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

# read inception features
def read_inception(fname):
    with open(fname) as f:
        for line in f:
            pairs=line.split()
            incept_temp = { int(p.split(':')[0]) : float(p.split(':')[1]) for p in pairs}
    
    incept = np.zeros(6075)
    for idx in incept_temp.keys():
        incept[idx-1] = incept_temp[idx]
    return incept



# spearman score
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

#Load Inception Features
df = pd.DataFrame(columns = ['video', 'arrayInfo'])


dir = '../features/InceptionV3/'

for filename in os.listdir(dir):
    if filename.endswith(".txt"):
        path = os.path.join(dir, filename)
        array = read_inception(path)
        if "-56" in filename:
          filename.replace('-56','')
          fileName= filename.replace(".txt",".webm")
          df = df.append({'video': fileName, 'arrayInfo': array}, ignore_index=True)
    else:
        break




count=0

#rename videos
for item in df['video']:
  df['video'][count]=item.replace('-56.webm','.webm')
  count = count + 1



# load the ground truth values
label_path = '../ground-truth/'
labels=pd.read_csv(label_path+'ground-truth_dev-set.csv')


df_inception = df.merge(labels,on=["video"],how="inner")



result_array = np.empty((0, 6075))

for line in df_inception['arrayInfo']:
    result_array = np.append(result_array, np.array([line]), axis=0)



arrayInfo = df_inception['arrayInfo'].values


X_arr = result_array
print(type(X_arr))
print(X_arr)




Y=df_inception[['short-term_memorability','long-term_memorability']].values  #targets
X=X_arr #input


#split randomly
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('Y_train', Y_train.shape)
print('Y_test', Y_test.shape)



n_cols = X_train.shape[1]



model = Sequential()
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
# Add the second layer
model.add(Dense(100, activation='relu'))
# Add the output layer
model.add(Dense(2))



# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
   



# Fit the model
history=model.fit(X_train, Y_train, validation_split=0.3, epochs=40)


predictions = model.predict(X_test)
print(predictions)
Get_score(predictions, Y_test) 

print('the short-term score is {:.3f}'.format(spearman_corr(Y_test[:,0],predictions[:,0])))
print('the long-term score is {:.3f}'.format(spearman_corr(Y_test[:,1],predictions[:,1])))






