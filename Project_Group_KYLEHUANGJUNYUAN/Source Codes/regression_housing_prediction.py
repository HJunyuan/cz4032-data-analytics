from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from subprocess import check_output
from math import sqrt
import math
import seaborn as sns

#print(check_output(["ls", "./input"]).decode("utf8"))

def dropped_redundunt_features(data):

    #find columns with high percentage of zeroes
    find_zeroes = data.columns[(data == 0).any()]
    num = (data[find_zeroes] == 0).sum()
    per_zeros = round(num/(data.shape[0]),2)
    
    #find features that has more than 80% zeroes
    features_with_zeroes = per_zeros[per_zeros>=0.80].index
    
    #Remove columns
    data.drop(features_with_zeroes,axis=1,inplace=True)

    #find columns with high percentage of NA
    find_na = data.isna().any()
    columns_na = data[data.columns[find_na]]
    na = columns_na.isna().sum()
    per_na = round(na/(data.shape[0]),2)

    #find features that has more than 80% NA
    features_with_NA = per_na[per_na>=0.80].index
    
    #Remove columns
    data.drop(features_with_NA,axis=1,inplace=True)

    return data

def extract_feature(col_names, data):

    data = data[col_names[1:]]
    return data


def feature_discovery(data,output):
    data = pd.concat([data,output],sort=False, axis=1)
    correlationMatrix = data.corr()

    redundunt = []
    for i in correlationMatrix:
        for j in correlationMatrix:
            if(i != j and i != 'SalePrice' and j != 'SalePrice'):
                if(correlationMatrix[i][j] >= 0.90):
                    redundunt.append(j)

    data.drop(redundunt,axis=1,inplace=True)
    new_correlationMatrix = data.corr()

    k = 80
    cols = new_correlationMatrix.nlargest(k, 'SalePrice')['SalePrice'].index

    #Print indexes with outlier
    # print(data.loc[data.GarageArea>3, ['SalePrice']])
    # print("=====================")
    # print(data.loc[data.GrLivArea>4, ['SalePrice']])
    # print("=====================")
    # print(data.loc[data.TotalBsmtSF>4, ['SalePrice']])

    return cols,k

def heatmapPlot(data,output,cols):
    data = pd.concat([data,output],sort=False, axis=1)
    #Plot Heatmap
    f, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(data[cols].corr(), vmax=.8, square=True)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()

def showIndividualCorrelation(data,output,feature_input, features):
    data = pd.concat([data,output],sort=False, axis=1)
    #Plot individual correlation
    x_axis = features[int(feature_input)]
    data.plot(x=x_axis,y='SalePrice',kind='scatter')
    plt.show()
    
def replaceNARows(data):

    #Find string and number types with NA
    na = data.isna().any()
    na_columns = data[data.columns[na]]
    numeric_info = na_columns._get_numeric_data().isna().sum()
    string_features = [x for x in na_columns if x not in na_columns._get_numeric_data().columns]
    string_info = data[string_features].isna().sum()
    print("========Numeric Features with NA=============\n")
    print(numeric_info)
    print("========Non-numeric Features with NA=========\n")
    print(string_info)
    print("\n")

    #Replace Numerical NA with average value
    data['LotFrontage'] = data['LotFrontage'].fillna(data['LotFrontage'].mean())
    data['MasVnrArea'] = data['MasVnrArea'].fillna(data['MasVnrArea'].mean())
    data['GarageYrBlt'] = data['GarageYrBlt'].fillna(0)

    #Replace String NA with 'None'
    data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode()[0])
    data['MasVnrType'] = data['MasVnrType'].fillna('None')
    data['BsmtQual'] = data['BsmtQual'].fillna('None')
    data['BsmtCond'] = data['BsmtCond'].fillna('None')
    data['BsmtExposure'] = data['BsmtExposure'].fillna('None')
    data['BsmtFinType1'] = data['BsmtFinType1'].fillna('None')
    data['BsmtFinType2'] = data['BsmtFinType2'].fillna('None')
    data['FireplaceQu'] = data['FireplaceQu'].fillna('None')
    data['GarageType'] = data['GarageType'].fillna('None')
    data['GarageFinish'] = data['GarageFinish'].fillna('None')
    data['GarageQual'] = data['GarageQual'].fillna('None')
    data['GarageCond'] = data['GarageCond'].fillna('None')

    return data

def rmsle(y, y_pred):
    #Root mean squared logartihmic error
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def plotLineScatterGraph(predicted_dataframe,test_output_dataframe):
    #line to scatter comparison
    x_ax = range(len(predicted_dataframe))
    plt.title('Predicted vs Actual for 200 Test Samples')
    plt.plot(x_ax, predicted_dataframe,label='Predicted')
    plt.scatter(x_ax, test_output_dataframe,color="red",s=5,label='Actual')
    plt.legend()

    plt.show()

def plotLinetoLineGraph(predicted_dataframe,test_output_dataframe):
    #line to line comparison
    ax = plt.gca()
    plt.title('Predicted vs Actual for 100 Test Samples')
    predicted_dataframe.plot(kind='line', x='number of rows', y='predicted', ax=ax, label='Predicted')
    test_output_dataframe['SalePrice'].plot(kind='line', x='number of rows', y='actual',color='red', ax=ax, label='Actual')
    plt.legend()

    plt.show()

def plotRegressionGraph(prediction,test_output):
    #Regression graph
    fig, ax = plt.subplots()
    plt.figure(1,figsize=(14,10))
    ax.scatter(prediction, test_output)
    ax.plot([prediction.min(),prediction.max()],[prediction.min(),prediction.max()], 'k--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    plt.show()

def plotBarGraph(prediction,test_output):
    #Bar Graph
    df = pd.DataFrame({'Actual': test_output,'Predicted': prediction})
    df1 = df.head(50)
    df1.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.xticks(rotation=45)
    plt.xlabel('Test Sample')
    plt.ylabel('SalePrice')
    plt.legend()

    plt.show()

#Retrive datasets
train_features = pd.read_csv('trainX.csv')
train_output = pd.read_csv('trainY.csv')

test_features = pd.read_csv('testX.csv')
test_output = pd.read_csv('testY.csv')

#One hot encode data
num_of_train = train_features.shape[0]
num_of_test = test_features.shape[0]

all_data = pd.concat([train_features,test_features]).reset_index(drop=True)

#Clean redudundant features
remove_redundant_data = dropped_redundunt_features(all_data)

#Replace NA values
replace_na_values = replaceNARows(remove_redundant_data)

#One hot encode data
encoded_data = pd.get_dummies(replace_na_values)

#Split data to train and test
train = encoded_data.iloc[:num_of_train,:]
test = encoded_data.iloc[num_of_train:,:]

#Get correlation after normalization
features, num_of_features= feature_discovery(train, train_output)

#Final train and test feature set used
train = extract_feature(features, train)
test = extract_feature(features, test)

while True:
    print("Show Heatmap?\n1.Yes\n2.No\n")
    heatmap = input("Choice: ")
    print("\n")
    if (int(heatmap) == 1 ):
        heatmapPlot(train, train_output, features)
    else:
        break

while True:
    print("Show Individual Correlation?\n1.Yes\n2.No\n")
    showcorrelation = input("Choice ")
    print("\n")
    if (int(showcorrelation) == 1):
        feature_input = input("Select one feature from the top %s feature: "%(str(num_of_features)))
        print("\n")
        if(int(feature_input) < int(num_of_features)):
            showIndividualCorrelation(train, train_output, feature_input, features)
        else: 
            print("Invalid feature input")
    else:
        break

#Remove Outlier
train.drop(axis=0, index=385,inplace=True)
train.drop(axis=0, index=753,inplace=True)
train_output.drop(axis=0, index=753,inplace=True)
train_output.drop(axis=0, index=385,inplace=True)

#Combine newly cleaned train and test for normalization
new_num_of_train = train.shape[0]
combine_for_normalization = pd.concat([train,test]).reset_index(drop=True)

#Normalize encoded data
normalized_data = (combine_for_normalization - np.mean(combine_for_normalization, axis=0))/np.std(combine_for_normalization,axis=0)

#Split into train and test again
train = normalized_data.iloc[:new_num_of_train,:]
test = normalized_data.iloc[new_num_of_train:,:]

#Define model
model1 = LinearRegression()
model2 = SVR(kernel='linear', C=1e3, gamma='auto')
model3 = Ridge()
model4 = RandomForestRegressor()
model5 = xgb.XGBRegressor(objective='reg:squarederror')
modelinput = ""

while True:

    print("Choose Regression Model:\n1.Linear Regression\n2.SVR\n3.Ridge\n4.RandomForest\n5.XGB\n")
    modelinput  = input("Choice:")
    print("\n")

    if (int(modelinput)==1):
        model = model1
        break
    elif (int(modelinput)==2):
        model = model2
        break
    elif (int(modelinput)==3):
        model = model3
        break
    elif (int(modelinput)==4):
        model = model4
        break
    elif (int(modelinput)==5):
        model = model5
        break
    else:
        print("Wrong Input.")

#Make Prediction 
train = np.array(train)
train_output = np.array(train_output)
test = np.array(test)
prediction = (model.fit(train, train_output.ravel())).predict(test)

if (int(modelinput)==3):
    prediction = prediction.flatten()

test_output_dataframe = test_output
test_output = test_output['SalePrice'].values
mse = mean_squared_error(test_output,prediction)
mae = mean_absolute_error(test_output, prediction)
msle = rmsle(test_output, prediction)
rmse = sqrt(mse)

print("Mean Square Error: "+str(mse))
print("Mean Absolute Error: "+str(mae))
print("Root Mean Square Logrithmic Error: "+str(msle))
print("Root Mean Square Error: "+str(rmse))
print("\n")

while True:

    try:
        print("Graph to plot.\n1.Regression\n2.Bar\n3.Line\n4.Scatter\n5.Exit\n")
        choice = input("Choice: ")
        print("\n")

        if(int(choice) == 1):
            #Regression Graph
            plotRegressionGraph(prediction,test_output)
        elif (int(choice) == 2):
            #Bar Graph
            plotBarGraph(prediction,test_output)
        elif(int(choice) == 3):
            #Line Graph 
            #Extract 100 rows from predicted and test
            predicted_dataframe = pd.Series(prediction)
            predicted_dataframe = predicted_dataframe.iloc[0:100]
            test_output_dataframe = test_output_dataframe.iloc[0:100]
            plotLinetoLineGraph(predicted_dataframe, test_output_dataframe)
        elif(int(choice) == 4):
            #Scatter Graph
            #Extract 100 rows from predicted and test
            predicted_dataframe = pd.Series(prediction)
            predicted_dataframe = predicted_dataframe.iloc[0:100]
            test_output_dataframe = test_output_dataframe.iloc[0:100]
            plotLineScatterGraph(predicted_dataframe, test_output_dataframe)
        else:
            print("Exiting...")
            break
    except KeyboardInterrupt:
        print("Exiting...")
        break

    
        






