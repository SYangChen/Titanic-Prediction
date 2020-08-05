#https://www.kaggle.com/c/titanic/data
import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras import optimizers
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv( 'train.csv' )

ageData = dict()
print('\n\t\t---Name---')
nameList = ['Mr.', 'Sir.', 'Dr.', 'Major.', 'Master.','Ms.', 'Miss.', 'Mrs.', 'Lady.']
for title in nameList:
    num = data[(data['Name'].str.contains(title))]['Name'].count()
    age = data[(data['Name'].str.contains(title))]['Age'].mean()
    print("{} –> {}, Age average is {:.2f}".format(title, num, age))
    ageData[title] = round(age,2)

temp = list( data['Age'] )
NAME = np.asarray( data['Name'] )

myList = []
for i in range( len(data) ):
    if str(temp[i]) != 'nan':
        myList.append(temp[i])
    else:
        for j in range( len(nameList) ):
            if nameList[j] in NAME[i]:
                myList.append(ageData[nameList[j]])
                break;
            else:
                myList.append(data['Age'].median())
                break;

df = pd.DataFrame({'Age':myList})
data = data.drop('Age', axis=1)
data = pd.concat( [df, data], axis = 1 )

for i in range ( len( data ) ) :
    if ( data['Ticket'][i] == 'LINE' ) :
        data = data.drop([i])


label = np.asarray( data['Survived'] )


data = data.drop( ['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin'], axis = 1 )
data['Family'] = data['Parch']+data['SibSp']
data = data.drop( ['Parch', 'SibSp'], axis = 1 )

one_hot_encoding = pd.get_dummies( data['Pclass'] )
one_hot_encoding = one_hot_encoding.rename( columns = { x:'Pclass-'+str(x) for x in one_hot_encoding.columns } )
data = data.drop( 'Pclass', axis = 1 )
data = pd.concat( [one_hot_encoding, data], axis = 1 )
#data['Pclass'] = data['Pclass'].astype( 'category' ).cat.codes

one_hot_encoding = pd.get_dummies( data['Sex'] )
data = data.drop( 'Sex', axis = 1 )
data = pd.concat( [one_hot_encoding, data], axis = 1 )
###data['Sex']=data['Sex'].map({'male':1,'female':2})
#data['Sex'] = data['Sex'].astype( 'category' ).cat.codes

data['Embarked'] = data['Embarked'].fillna( 'S' )
one_hot_encoding = pd.get_dummies( data['Embarked'] )
data = data.drop( 'Embarked', axis = 1 )
data = pd.concat( [one_hot_encoding, data], axis = 1 )
#data['Embarked'] = data['Embarked'].astype( 'category' ).cat.codes

#data['Age'] = data['Age'].fillna( data['Age'].median() )
data['Fare'] = data['Fare'].fillna( data['Fare'].mean() )

'''
plt.figure(figsize=(14,12))
correlation_matrix = data.corr()
axes_obj = sns.heatmap(correlation_matrix, annot=True)
'''
#%% upload to kaggle
#'''
testData = pd.read_csv( 'test.csv' )

ageData = dict()
print('\n\t\t---Name---')
nameList = ['Mr.', 'Sir.', 'Dr.', 'Major.', 'Master.','Ms.', 'Miss.', 'Mrs.', 'Lady.']
for title in nameList:
    num = testData[(testData['Name'].str.contains(title))]['Name'].count()
    age = testData[(testData['Name'].str.contains(title))]['Age'].mean()
    print("{} –> {}, Age average is {:.2f}".format(title, num, age))
    ageData[title] = round(age,2)

temp = list( testData['Age'] )
NAME = np.asarray( testData['Name'] )

myList = []
for i in range( len(testData) ):
    if str(temp[i]) != 'nan':
        myList.append(temp[i])
    else:
        for j in range( len(nameList) ):
            if nameList[j] in NAME[i]:
                myList.append(ageData[nameList[j]])
                break;
            else:
                myList.append(testData['Age'].median())
                break;

df = pd.DataFrame({'Age':myList})
testData = testData.drop('Age', axis=1)
testData = pd.concat( [df, testData], axis = 1 )

for i in range ( len( testData ) ) :
    if ( testData['Ticket'][i] == 'LINE' ) :
        testData = testData.drop([i])

pid = np.asarray( testData['PassengerId'] )
testData = testData.drop( ['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1 )
testData['Family'] = testData['Parch']+testData['SibSp']
testData = testData.drop( ['Parch', 'SibSp'], axis = 1 )

one_hot_encoding = pd.get_dummies( testData['Pclass'] )
one_hot_encoding = one_hot_encoding.rename( columns = { x:'Pclass-'+str(x) for x in one_hot_encoding.columns } )
testData = testData.drop( 'Pclass', axis = 1 )
testData = pd.concat( [one_hot_encoding, testData], axis = 1 )
#testData['Pclass'] = testData['Pclass'].astype( 'category' ).cat.codes

one_hot_encoding = pd.get_dummies( testData['Sex'] )
testData = testData.drop( 'Sex', axis = 1 )
testData = pd.concat( [one_hot_encoding, testData], axis = 1 )
###testData['Sex']=testData['Sex'].map({'male':1,'female':2})
#testData['Sex'] = testData['Sex'].astype( 'category' ).cat.codes

testData['Embarked'] = testData['Embarked'].fillna( 'S' )
one_hot_encoding = pd.get_dummies( testData['Embarked'] )
testData = testData.drop( 'Embarked', axis = 1 )
testData = pd.concat( [one_hot_encoding, testData], axis = 1 )
#testData['Embarked'] = testData['Embarked'].astype( 'category' ).cat.codes

#testData['Age'] = testData['Age'].fillna( testData['Age'].median() )
testData['Fare'] = testData['Fare'].fillna( testData['Fare'].mean() )


#%%
X = np.asarray( data )
Y = np_utils.to_categorical( label )

# bound = int(data.shape[0]*0.2)
# bound *= -1

xTrain = X
xTest = np.asarray( testData )
yTrain = Y
# yTest = Y[bound:]
#''' # upload to kaggle
#%%
'''
X = np.asarray( data )
Y = np_utils.to_categorical( label )

bound = int(data.shape[0]*0.2)
bound *= -1

xTrain = X[:bound]
xTest = X[bound:]
yTrain = Y[:bound]
yTest = Y[bound:]
'''
#%%
model = Sequential()
model.add( Dense( units = 12, input_dim = X.shape[1], kernel_initializer = 'normal', activation = 'elu') )
model.add( Dense( units = 144, kernel_initializer = 'normal', activation = 'elu' ) )
model.add( Dense( units = 96, kernel_initializer = 'normal', activation = 'elu' ) )
model.add( Dense( units = 24, kernel_initializer = 'normal', activation = 'elu' ) )
model.add( Dense( units = 12, kernel_initializer = 'normal', activation = 'elu' ) )
model.add( Dense( units = 2, kernel_initializer = 'normal', activation = 'softmax' ) )

#my_opt = optimizers.Adam(lr=0.001, decay=0.00)
model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'] )
model.fit( x = xTrain, y = yTrain, validation_split = 0.2, epochs = 200, batch_size = 24, verbose = 2 )
#%% upload to kaggle
#'''
predictions = model.predict_classes( xTest )
print( predictions )
dataset = pd.DataFrame( { 'PassengerID': pid, 'Survived': predictions }, columns = ['PassengerID', 'Survived'] )
dataset.to_csv('result.csv', index = 0)
#''' # upload to kaggle
#%%
'''
loss,acc = model.evaluate( xTest, yTest )
print( "\n[Info] Loss value of testing data : {:f}".format(loss) )
print( "[Info] Accuracy of testing data : {:2.1f}%".format(acc*100.0) )
print()
predictions = model.predict_classes( xTest )
print( 'Prediction:', predictions[0:10] )
print( 'Answer:    ', label[bound:][0:10] )
'''