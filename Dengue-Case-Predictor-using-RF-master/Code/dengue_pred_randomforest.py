import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

x = pd.read_csv('C:\\Users\\Agrim Nautiyal\\Desktop\\data _drive\\dengue_features_train.csv')
y = pd.read_csv('C:\\Users\\Agrim Nautiyal\\Desktop\\data _drive\\dengue_labels_train.csv')
x.drop(['city', 'week_start_date'], axis=1, inplace=True)
y= y['total_cases']
y=pd.DataFrame(y)

x = x.fillna(x.mean())
y = y.fillna(y.mean())

#now that we have our data as features and labels we can now create a split between test and train data with a random split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)
clf= DecisionTreeClassifier()
clf.fit(x_train, y_train)
regr = RandomForestRegressor()
regr.fit(x_train, y_train)
y_pred1 = clf.predict(x_test)
print(mean_squared_error(y_test, y_pred1))
print("Accuracy of decision tree model : " + str(clf.score(x,y)*100) + str("%"))

#NOTE THAT IN REGRESSIONAL ANALYSIS THE MSE WAS AROUND 2033 and accuracy of about 17.5%. This is a positive indicator that a non customised decision tree classifier works better than regression
#for this particular dataset as accuracy jumps from 17.5 % to 72.25%
#lets try to bring the mse even lower

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor()
regr.fit(x_train, y_train)

y_pred2 = regr.predict(x_test)
print(mean_squared_error(y_test, y_pred2))
print("Accuracy of random forest regressor : "+ str(regr.score(x,y)*100) + str("%"))

#accuracy of random forest regressor is brought to a whooping 81.1% 
#Remeber, we started with 17.5% and 2033 MSE from regressional analysis <-


model_test = RandomForestRegressor()

def sigmoid(x):
    return(  (1.0)/(1+np.exp(-1*x)  ) )
for i in range(10):
    x_train_, x_test_, y_train_, y_test_ = train_test_split(x,y,test_size =sigmoid(i)*0.1, random_state = i )
    model_test.fit(x_train_, y_train_)

#we use sigmoid to introduce variation in train and test params and bring value to  a range from 0.0 to 1.0

    y_pred_ = model_test.predict(x_test)
    print(model_test.score(x,y)*100)
    # output of the above snippet was  : 91.08756165226092

    # 4- 63; 10- 88. 100- 87, 50- 81, 10-87, 10-90
print(mean_squared_error(y_test, y_pred_))
model_test.get_params()


#This way, we train our model recursively on the training targets and features and make our model recursively better 
#since we are splitting the data here as training and testing data and repeatedly training it there's a high chance, 
#that the model will cover all the training points and this can lead to a severe case of
''' OVERFITTING'''
'''
We Can prevent overfitting by keeping the number of iterations in the above for loop less. This will lower the probablity of training 
data to be memorised and hence even though we get a high accuracy on the training data, we cannot be sure that  we will also get
a high accuracy on data that the model has never seen before. 
This helps us conclude  that without any major customisation , for the given dataset we can expect the best built in model as 
RandomForestRegressor with an accuracy in the range of 75-85 % with some amount of iterative learning (but not too much)
which compared to plain regression gives us a much better margin 
'''


import matplotlib.pyplot as plt


#from the below plots it is visible that the model fits the base values really well and unlike regression, also stretches towards
#the upper curve, to try to fit the data it has never seen



#depciting extent of model fitting on attribute : 'weekofyear' of x_test VS predictions(blue) and actual value(red) 

plt.scatter(x_test['weekofyear'], y_pred1, color ='blue')
plt.scatter(x_test['weekofyear'], y_test, color = 'red')
plt.show()


#model fitting in another attribute of x_test : precipitation_amt_mm : 

plt.scatter(x_test['precipitation_amt_mm'], y_pred1, color ='yellow')
plt.scatter(x_test['precipitation_amt_mm'], y_test, color = 'green')
plt.show()


#plotting reanalysis_specific_humidity_g_per_kg attbt. of x_test 
plt.scatter(x_test['reanalysis_specific_humidity_g_per_kg'],y_pred1, color = 'blue')
plt.scatter(x_test['reanalysis_specific_humidity_g_per_kg'], y_test, color = 'green')
plt.show()
#blue - predicted values . 
#green - actual values 

# plotting reanalysis_max_air_temp_k and reanalysis_min_air_temp_k : 

plt.scatter(x_test['reanalysis_max_air_temp_k'], y_pred1, color = 'red')
plt.scatter(x_test['reanalysis_min_air_temp_k'], y_pred1, color = 'red')
plt.scatter(x_test['reanalysis_max_air_temp_k'], y_test, color = 'yellow')
plt.scatter(x_test['reanalysis_min_air_temp_k'], y_test, color = 'blue')
plt.show()



plt.scatter(x_test['ndvi_ne'], y_test, color = 'red')
plt.scatter(x_test['ndvi_nw'], y_test, color = 'red')
plt.scatter(x_test['ndvi_se'], y_test, color = 'red')
plt.scatter(x_test['ndvi_sw'], y_test, color = 'red')
plt.scatter(x_test['reanalysis_air_temp_k'], y_test, color = 'red')
plt.scatter(x_test['reanalysis_avg_temp_k'], y_test, color = 'red')
plt.scatter(x_test['reanalysis_avg_temp_k'], y_test, color = 'red')
plt.scatter(x_test['reanalysis_dew_point_temp_k'], y_test, color = 'red')
plt.scatter(x_test['station_avg_temp_c'], y_test, color = 'red')

plt.scatter(x_test['ndvi_ne'], y_pred1, color = 'yellow')
plt.scatter(x_test['ndvi_nw'], y_pred1, color = 'yellow')
plt.scatter(x_test['ndvi_se'], y_pred1, color = 'yellow')
plt.scatter(x_test['ndvi_sw'], y_pred1, color = 'yellow')
plt.scatter(x_test['reanalysis_air_temp_k'],y_pred1, color = 'green')
plt.scatter(x_test['reanalysis_avg_temp_k'], y_pred1, color = 'green')
plt.scatter(x_test['reanalysis_avg_temp_k'], y_pred1, color = 'green')
plt.scatter(x_test['reanalysis_dew_point_temp_k'], y_pred1, color = 'green')
plt.scatter(x_test['station_avg_temp_c'], y_pred1, color = 'green')
#   red signifies actual values, yellow and green signify prediction points


# now we will visualise onlt the final plot with respect to the relatively over fitted model that we created as model_test

plt.scatter(x_test['ndvi_ne'], y_test, color = 'red')
plt.scatter(x_test['ndvi_nw'], y_test, color = 'red')
plt.scatter(x_test['ndvi_se'], y_test, color = 'red')
plt.scatter(x_test['ndvi_sw'], y_test, color = 'red')
plt.scatter(x_test['reanalysis_air_temp_k'], y_test, color = 'red')
plt.scatter(x_test['reanalysis_avg_temp_k'], y_test, color = 'red')
plt.scatter(x_test['reanalysis_avg_temp_k'], y_test, color = 'red')
plt.scatter(x_test['reanalysis_dew_point_temp_k'], y_test, color = 'red')
plt.scatter(x_test['station_avg_temp_c'], y_test, color = 'red')

plt.scatter(x_test['ndvi_ne'], y_pred_, color = 'yellow')
plt.scatter(x_test['ndvi_nw'], y_pred_, color = 'yellow')
plt.scatter(x_test['ndvi_se'], y_pred_, color = 'yellow')
plt.scatter(x_test['ndvi_sw'], y_pred_, color = 'yellow')
plt.scatter(x_test['reanalysis_air_temp_k'],y_pred_, color = 'green')
plt.scatter(x_test['reanalysis_avg_temp_k'], y_pred_, color = 'green')
plt.scatter(x_test['reanalysis_avg_temp_k'], y_pred_, color = 'green')
plt.scatter(x_test['reanalysis_dew_point_temp_k'], y_pred_, color = 'green')
plt.scatter(x_test['station_avg_temp_c'], y_pred_, color = 'green')
#   red signifies actual values, yellow and green signify prediction points


# as we see from the above  plot, comparing to the plot before this model fits the training data really well, and has better reach 
#on the points towards the higher access as compared to the prev plot
#though this denotes a high accuracy (91 %) on the training set,  we cannot predict if this same accuracy will be maintained
# on data that model_test model has never seen.


# as we have demonstrated a bette model for the dengue data set compared to reression, and gotten an idea of what overfitting is
#and how it can occur, we can now look at ways of preventing overfitting and creating a model better trained over multiple iterations just the way
# tried above.
final_error = 10000000000000
#let us now try parameter tuning without avoiding any over fitting
#for this we'll have to create a costly model selection method (time cost) as we'll iterate model over lists of just 3 params with 6 samples each and select the best combo 
#from all obtained mean squared error scores 
final_params = []
a = [[[None for i in range(6)] for j in range(6)] for k in range(6)]


n_est = range(1,100)
max_feat = [ 'sqrt', 'log2', 'auto']
min_sample_leaf_ = range(1,3)

#now we start training the model over all the above params and test the score
#the lowest error gets appended and list of params get appended to final_params= []
final_error = -10000000000000
for i in n_est:
    for j in max_feat:
        for k in min_sample_leaf_ :
            model = RandomForestRegressor(n_estimators = i, max_features = j, min_samples_leaf = k, criterion ='mse', min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True)
            #we have now created our model 
            model.fit(x_train , y_train)
            y_pred_test = model.predict(x_test)
            m= model.score(x,y)
            if m>final_error:
                final_error = m
                final_params =[]
                final_params.extend((i,j,k))
print(m)
print(final_params)



m = (RandomForestRegressor(n_estimators = 40, max_features = 'auto', min_samples_leaf = 1, criterion ='mse', min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, min_samples_split = 2, oob_score = 'False'))


m.fit(x_train, y_train)
y_pr = m.predict(x_test)
print(mean_squared_error(y_test, y_pr))
print(m.score(x,y))

#out put was : 923.47801201373
#              0.8158599138683817

#now with some parameter tuning we have managed to bring accuracy of our model from 17% in regression to 79% of a raw model with mse of over 1000, to now 81% with 
#MSE of 923.478 
#let us now try to make our predictions and see what happens

X = pd.read_csv('C:\\Users\\Agrim Nautiyal\\Desktop\\data _drive\\dengue_features_test.csv')
X.drop(['city', 'week_start_date'], axis=1, inplace=True)
X = X.fillna(x.mean())

Y = m.predict(X)
#prediction = pd.DataFrame(Y, columns=['predictions']).to_csv('prediction.csv')
prediction = pd.DataFrame(Y, columns=['predictions']).round()

prediction = prediction.astype(dtype = 'int32')
pred = prediction.to_csv('predict.csv')

