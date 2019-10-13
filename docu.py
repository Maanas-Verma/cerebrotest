'''Branch EPH
Name Maanas Verma
En no. 18122011
'''

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


#the above code will provide the formation like give down

'''
#                            OLS Regression Results                            
#==============================================================================
#Dep. Variable:                      y   R-squared:                       0.004
#Model:                            OLS   Adj. R-squared:                 -0.001
#Method:                 Least Squares   F-statistic:                    0.7771
#Date:                Sun, 13 Oct 2019   Prob (F-statistic):              0.820
#Time:                        03:30:22   Log-Likelihood:                -5170.9
#No. Observations:                7192   AIC:                         1.041e+04
#Df Residuals:                    7157   BIC:                         1.065e+04
#Df Model:                          34                                         
#Covariance Type:            nonrobust                                         
#==============================================================================
#                 coef    std err          t      P>|t|      [0.025      0.975]
#------------------------------------------------------------------------------
#const          0.1796      0.046      3.876      0.000       0.089       0.271
#x1          4.797e-07   2.83e-06      0.169      0.865   -5.07e-06    6.03e-06
#x2          2.946e-06   1.24e-06      2.374      0.018    5.14e-07    5.38e-06
#x3          2.946e-06   1.24e-06      2.374      0.018    5.14e-07    5.38e-06
#x4             0.0028      0.004      0.698      0.485      -0.005       0.011
#x5            -0.0055      0.004     -1.378      0.168      -0.013       0.002
#x6            -0.0021      0.004     -0.520      0.603      -0.010       0.006
#x7            -0.0022      0.004     -0.529      0.597      -0.010       0.006
#x8            -0.0052      0.004     -1.286      0.199      -0.013       0.003
#x9            -0.0011      0.004     -0.276      0.783      -0.009       0.007
#x10            0.0046      0.004      1.121      0.262      -0.003       0.013
#x11            0.0003      0.004      0.076      0.939      -0.008       0.008
#x12           -0.0008      0.004     -0.201      0.841      -0.009       0.007
#x13           -0.0030      0.004     -0.732      0.464      -0.011       0.005
#x14           -0.0024      0.004     -0.580      0.562      -0.010       0.006
#x15           -0.0046      0.004     -1.121      0.262      -0.013       0.003
#x16           -0.0001      0.000     -0.741      0.459      -0.000       0.000
#x17        -2.595e-05      0.000     -0.193      0.847      -0.000       0.000
#x18         7.808e-05      0.000      0.570      0.568      -0.000       0.000
#x19           -0.0002      0.000     -1.732      0.083      -0.000    3.09e-05
#x20        -1.995e-05      0.000     -0.147      0.883      -0.000       0.000
#x21            0.0127      0.007      1.769      0.077      -0.001       0.027
#x22           -0.0007      0.007     -0.097      0.923      -0.015       0.014
#x23           -0.0018      0.007     -0.251      0.802      -0.016       0.012
#x24           -0.0029      0.007     -0.411      0.681      -0.017       0.011
#x25            0.0013      0.007      0.181      0.856      -0.013       0.015
#x26            0.0026      0.007      0.355      0.723      -0.012       0.017
#x27           -0.0045      0.007     -0.625      0.532      -0.019       0.010
#x28           -0.0022      0.007     -0.313      0.755      -0.016       0.012
#x29           -0.0055      0.007     -0.774      0.439      -0.020       0.008
#x30           -0.0084      0.007     -1.159      0.246      -0.023       0.006
#x31           -0.0026      0.007     -0.359      0.720      -0.017       0.011
#x32           -0.0020      0.007     -0.281      0.779      -0.016       0.012
#x33            0.0048      0.007      0.670      0.503      -0.009       0.019
#x34            0.0093      0.007      1.291      0.197      -0.005       0.023
#x35           -0.0036      0.007     -0.511      0.609      -0.018       0.010
#==============================================================================
#Omnibus:                        5.886   Durbin-Watson:                   2.006
#Prob(Omnibus):                  0.053   Jarque-Bera (JB):                5.896
#Skew:                           0.062   Prob(JB):                       0.0524
#Kurtosis:                       2.936   Cond. No.                     2.25e+15
#==============================================================================
'''


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
NN_model.summary()   #this code will show the summry of our devloped neural network layers as shown below

'''
#Model: "sequential_24"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#dense_127 (Dense)            (None, 4)                 20        
#_________________________________________________________________
#dense_128 (Dense)            (None, 300)               1500      
#_________________________________________________________________
#dense_129 (Dense)            (None, 200)               60200     
#_________________________________________________________________
#dense_130 (Dense)            (None, 100)               20100     
#_________________________________________________________________
#dense_131 (Dense)            (None, 1)                 101       
#=================================================================
#Total params: 81,921
#Trainable params: 81,921
#Non-trainable params: 0
#_________________________________________________________________
'''

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

