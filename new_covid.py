#%%
# Import modules
from tensorflow.keras import Sequential,Input
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.layers import LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os,datetime
#%%
# Get data path
dfpath = os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
# load data
df = pd.read_csv(dfpath)
 
#%%
# Data inspection
df.info()
#%%
df.head()
#%%
df.describe()
#%%
#Convert to numbers
df['cases_new'] = pd.to_numeric(df['cases_new'], errors = 'coerce')
#%%
#Check NaN value
print(df['cases_new'].isna().sum())
#%%
plt.figure(figsize=(10,10))
plt.plot(df['cases_new'])
plt.show()
#%%
#Interpolating NaN value
df['cases_new'] = df['cases_new'].interpolate(method='polynomial',order=2)
#%%
plt.figure(figsize=(10,10))
plt.plot(df['cases_new'])
plt.show()
#%%
# Data pre-processing(Normalization)
data = df['cases_new'].values
data = data[::,None]
#%%
#Min-max Scaler
mm_scaler = MinMaxScaler()
mm_scaler.fit(data)
data = mm_scaler.transform(data)
#%%
#Window size = 30 days
win_size = 30
X_train = [] # a list 
Y_train = [] # a list

for i in range(win_size,len(data)):
    X_train.append(data[i-win_size:i])
    Y_train.append(data[i])

# Convert into numpy array
X_train = np.array(X_train)
Y_train = np.array(Y_train)
#%%
#Train-test split
X_train,X_test,Y_train,Y_test = train_test_split(X_train,Y_train,random_state=123)

#%%
# Model development
model = Sequential()
model.add(Input(shape=(30,1)))
model.add(LSTM(32))
model.add(Dropout(0.3))
model.add(Dense(1, activation = 'linear'))
model.summary()


#%%
# Model flow
from tensorflow.keras.utils import plot_model
plot_model(model,show_shapes=True) 
#%%
#Compile
model.compile(optimizer='adam',loss ='mse',metrics=['mse','mape'])
LOGS_PATH = os.path.join(os.getcwd(),'logs', datetime.datetime.now().strftime('%Y&m%d-%H%M%S'))
#%%
#Tensorboard
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH)
early_stop_callback = EarlyStopping(monitor='val_loss',patience=5)
hist = model.fit(X_train,Y_train,epochs=200,callbacks=[tensorboard_callback,early_stop_callback],validation_data=(X_test,Y_test))
#%%
# Load testing dataset path
test_csv = os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
df_test = pd.read_csv(test_csv)

#%%
#Convert to number
df_test['cases_new'] = pd.to_numeric(df_test['cases_new'], errors = 'coerce')
#%%
#Check NaN value
print(df_test['cases_new'].isna().sum())
#%%
plt.figure(figsize=(10,10))
plt.plot(df_test['cases_new'])
plt.show()
#%%
#Interpolate NaN value
df_test['cases_new'] = df_test['cases_new'].interpolate(method='polynomial',order=2)
#%%
plt.figure(figsize=(10,10))
plt.plot(df_test['cases_new'])
plt.show()


# %%
#Concatenate df
concat = pd.concat((df['cases_new'],df_test['cases_new']))
concat = concat[len(concat)-win_size-len(df_test):]

concat = mm_scaler.transform(concat[::,None])

X_testtest = []
Y_testtest = []

for i in range(win_size,len(concat)):
    X_testtest.append(concat[i-win_size:i])
    Y_testtest.append(concat[i])

X_testtest = np.array(X_testtest) 
Y_testtest = np.array(Y_testtest)

# to predict new cases based on the testing dataset

predicted_newcases = model.predict(X_testtest)

#%%
#Transform scale
Y_testtest = mm_scaler.inverse_transform(Y_testtest)
predicted_newcases = mm_scaler.inverse_transform(predicted_newcases)

# %%
#Plot graph
plt.figure()
plt.plot(Y_testtest,color='red')
plt.plot(predicted_newcases,color='blue')
plt.legend(['Actual','Predicted'])
plt.xlabel('time')
plt.ylabel('Cases')
plt.show()



# %%
#Model Evaluation
mape = mean_absolute_percentage_error(Y_testtest,predicted_newcases)
print(f'Mean absolute percentage error : {mape}') 
#%%
# Model saving
# save deep learning model

model.save('text.h5')

# %%
