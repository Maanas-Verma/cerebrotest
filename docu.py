#The very first work is of importing libraries that will going to be use in making model
#main libraries use in this documentation is keras for buiding an ann model


from keras.callbacks import ModelCheckpoint
from keras.models import Sequential             # for build up ann model
from keras.layers import Dense
import pandas as pd                             #for importing .csv files in code
import numpy as np
import statsmodels.api as sm
import warnings 
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)



# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X=np.append(arr=np.ones((7192,1)).astype(int),values=X,axis=1)    # this line of code is for adding a column intially of element of 1's


X_opt=X[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]]  #it is the very first assumption of X_opt (optimal features from X)

#now i am going to apply back elimination one by one of every features by taking p-value into account 



X_opt_1=np.append(arr=np.ones((7192,1)).astype(int),values=X_opt,axis=1)
regressor_OLS=sm.OLS(endog=y,exog=X_opt_1).fit()
regressor_OLS.summary() 

#now we eliminate feature that has maximum value of p-value 
#now we chage value of X_opt to X[:,[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]]  by eliminating 11'th feature of dataset
# then the below codes are showing direction of my back elimination

X_opt=X[:,[1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35]] #11,22
X_opt=X[:,[2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30,31,32,33,34,35]] #11,1,22
X_opt=X[:,[2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,21,23,24,25,26,27,28,29,30,31,32,33,34,35]] #11,1,22,20
X_opt=X[:,[2,3,4,5,6,7,8,9,10,12,13,14,15,16,17,18,19,21,23,24,26,27,28,29,30,31,32,33,34,35]] #11,1,22,20,25
X_opt=X[:,[2,3,4,5,6,7,8,9,10,12,13,14,15,16,18,19,21,23,24,26,27,28,29,30,31,32,33,34,35]] #11,1,22,20,25,17
X_opt=X[:,[2,3,4,5,6,7,8,9,10,12,13,14,15,16,18,19,21,24,26,27,28,29,30,31,32,33,34,35]] #11,1,22,20,25,17,23
X_opt=X[:,[2,3,4,5,6,7,8,9,10,13,14,15,16,18,19,21,24,26,27,28,29,30,31,32,33,34,35]] #11,1,22,20,25,17,23,12
X_opt=X[:,[2,3,4,5,6,7,8,10,13,14,15,16,18,19,21,24,26,27,28,29,30,31,32,33,34,35]] #11,1,22,20,25,17,23,12,9
X_opt=X[:,[2,3,4,5,6,7,8,10,13,14,15,16,18,19,21,24,26,27,28,29,30,31,33,34,35]] #11,1,22,20,25,17,23,12,9,32
X_opt=X[:,[2,3,4,5,6,7,8,10,13,14,15,16,18,19,21,24,26,27,29,30,31,33,34,35]] #11,1,22,20,25,17,23,12,9,32,28
X_opt=X[:,[2,3,4,5,6,7,8,10,13,14,15,16,18,19,21,24,27,29,30,31,33,34,35]] #11,1,22,20,25,17,23,12,9,32,28,26
X_opt=X[:,[2,3,4,5,6,7,8,10,13,14,15,16,18,19,21,24,27,29,30,33,34,35]] #11,1,22,20,25,17,23,12,9,32,28,26,31
X_opt=X[:,[2,3,4,5,6,7,8,10,13,14,15,16,18,19,21,27,29,30,33,34,35]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24
X_opt=X[:,[2,3,4,5,6,8,10,13,14,15,16,18,19,21,27,29,30,33,34,35]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7
X_opt=X[:,[2,3,4,5,8,10,13,14,15,16,18,19,21,27,29,30,33,34,35]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6
X_opt=X[:,[2,3,4,5,8,10,13,14,15,16,18,19,21,27,29,30,33,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35
X_opt=X[:,[2,3,4,5,8,10,13,15,16,18,19,21,27,29,30,33,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14
X_opt=X[:,[2,3,4,5,8,10,13,15,16,19,21,27,29,30,33,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18
X_opt=X[:,[2,3,4,5,8,10,13,15,16,19,21,29,30,33,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27
X_opt=X[:,[2,3,4,5,8,10,13,15,16,19,21,29,30,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33
X_opt=X[:,[2,3,5,8,10,13,15,16,19,21,29,30,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33,4
X_opt=X[:,[2,3,5,8,10,13,15,19,21,29,30,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33,4,16
X_opt=X[:,[2,3,5,8,10,15,19,21,29,30,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33,4,16,13
X_opt=X[:,[2,3,5,8,10,15,19,21,30,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33,4,16,13,29
X_opt=X[:,[2,3,5,8,10,15,19,21,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33,4,16,13,29,30
X_opt=X[:,[2,3,5,8,10,19,21,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33,4,16,13,29,30,15
X_opt=X[:,[2,3,5,8,19,21,34]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33,4,16,13,29,30,15,10
X_opt=X[:,[2,3,5,8,19,21]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33,4,16,13,29,30,15,10,34
X_opt=X[:,[2,3,5,19,21]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33,4,16,13,29,30,15,10,34,8
X_opt=X[:,[2,3,19,21]] #11,1,22,20,25,17,23,12,9,32,28,26,31,24,7,6,35,14,18,27,33,4,16,13,29,30,15,10,34,8,5


#for the last four features the p-vales are given below which are less than 0.01
# 	coef 	std err 	t 	P>|t| 	[0.025 	0.975]
#const 	0.1519 	0.016 	9.293 	0.000 	0.120 	0.184
#x1 	2.899e-06 	1.24e-06 	2.344 	0.019 	4.75e-07 	5.32e-06
#x2 	2.899e-06 	1.24e-06 	2.344 	0.019 	4.75e-07 	5.32e-06
#x3 	-0.0002 	0.000 	-1.673 	0.094 	-0.000 	3.87e-05
#x4 	0.0130 	0.007 	1.809 	0.070 	-0.001 	0.027

#now i make a ann model for solving this problem with taking only these four features

#the very first step is to have a class of Sequential from keras.model library
NN_model = Sequential()

#the second step is initialize it by 4 features with relu activation
NN_model.add(Dense(4, kernel_initializer='normal',input_dim = X_opt.shape[1], activation='relu'))


#third part is for creation of inner layers with tanh activation funtion for heighier accuracy
NN_model.add(Dense(300, kernel_initializer='normal',activation='tanh'))   #100
NN_model.add(Dense(200, kernel_initializer='normal',activation='tanh'))   #50
NN_model.add(Dense(100, kernel_initializer='normal',activation='tanh'))   #50

then for using ann for regression we have use linear activation in last layer for accurate and better output scaling
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

#now we are compiling our model for mean_squared_error for fullfill the score of competition
NN_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
NN_model.summary()   #this code will show the summry of our devloped neural network layers 

#the line given below will start the process of learning the data with every bach size 32 in a time 
#it runs for 500 times and crossvalidation part will randomlly slected from 0.4 times of given dataset
NN_model.fit(X_opt, y, epochs=500, batch_size=32, validation_split = 0.4, callbacks=callbacks_list)


#the function below is for creating a .csv file and save in same directory
def make_submission(prediction, sub_name):
  my_submission = pd.DataFrame({'id':pd.read_csv('test.csv').id,'target':prediction})
  my_submission.to_csv('{}.csv'.format(sub_name),index=True)
  print('A submission file has been made')
  

#below lines of code use to import test set and predict y from it and calling of make_submission() with name writtern in it
dataset_t = pd.read_csv('test.csv')
dataset_t = np.append(arr=np.ones((1000,1)).astype(int),values=dataset_t,axis=1)
data_t=dataset_t[:,[2,3,19,21]]  

predictions = NN_model.predict(data_t)

make_submission(predictions[:,0],'submission(anntanh).csv')

