import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import math

# data to use from the X_train.csv file
data = pd.read_csv('./X_train.csv')

# data clean up, removing the big amount of uneccessary 0.0
dataClean = data[(data['x_1'] != 0.0) | (data['y_1'] != 0.0) |
            (data['x_2'] != 0.0) | (data['y_2'] != 0.0) |
            (data['x_3'] != 0.0) | (data['y_3'] != 0.0)]

# comecaIndex is a stack with all the initial positions (where t == 0.0)
# since the uneccessary 0.0 were deleted, this is grouping only the initial positions
comecaIndex = np.hstack((dataClean[dataClean.t == 0.0].index.values))

# acabaIndex is a stack with all the final positions (where t == 0.0, but - 1 because this way is getting the row before the initial positions, that corresponds to the final positions)
# in the final is adding the last line, because the last final positions does not have in front an initial position
# since the uneccessary 0.0 were deleted, this is grouping only the initial positions
acabaIndex = np.hstack((dataClean[dataClean.t == 0.0].index.values - 1, dataClean.iloc[-1]['Id']))

# add to dataClean the initial positions corresponding to each position from each iteration
# add the initial positions from where the iteration starts until where the iteration ends
dataClean.loc[:, ['x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position']] = 0.0
#completed_data = pd.DataFrame()
for j in range(comecaIndex.size):
    dataClean.loc[comecaIndex[j]:acabaIndex[j+1], ['x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position']] = data.loc[comecaIndex[j], ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values 

# drop the velocities, since it's not being used
dataClean = dataClean.drop(columns=['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3'], inplace=False)
# it's not necessary to have the initial positions
# the solution wants to predict positions where t > 0.0
dataClean = dataClean[(dataClean['t'] != 0.0)]
# asserting the Id column
colunaId = dataClean.pop('Id')
dataClean.insert(0, 'Id', colunaId)

# 80% to train, 20% to val
train, val = train_test_split(dataClean, test_size=0.2, random_state=42)

# get the X_test and change the name of the columns
test = pd.read_csv('./X_test.csv')
cols = ['Id','t','x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position']
test.columns=cols

###     END OF TASK 1.1     ###

# features are the columns of entrance, to get the final positions, depending on the time and initial positions (that are the features)
featuresX = ['x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position', 't']

# y are the final positions, calculated depending on the initial positions and time
y = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

# assigning the inputs and outputs
entryTrain = train[featuresX]
outputTrain = train[y]
entryVal = val[featuresX]
outputVal = val[y]

entryTest  = test[featuresX]

# Pipeline with StandardScaler and LinearRegression
pipeline = Pipeline([('scaler', StandardScaler()),
             ('linear_regression', LinearRegression())])

pipeline.fit(entryTrain, outputTrain)

outputTrainPrediction = pipeline.predict(entryTrain)
outputValPrediction = pipeline.predict(entryVal)

outputTestPrediction = pipeline.predict(entryTest)

columns = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
# create a DataFrame from the predicted values
df = pd.DataFrame(outputTestPrediction, columns=columns)
# add a column for the 'id' and reorder the columns to have the output file (submission file) in the expected structure
df['id'] = df.index
df = df[['id', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]

# extract the submission file
outputPath = ('./baseline-model.csv')
df.to_csv(outputPath,index=False)

# calculate rmse values 
rmse_train =  math.sqrt(mean_squared_error(outputTrain, outputTrainPrediction))
rmse_val = math.sqrt(mean_squared_error(outputVal, outputValPrediction))

# plot
def plot_y_yhat(y_val,y_pred, plot_title = "plot"):
    labels = ['x_1','y_1','x_2','y_2','x_3','y_3']
    MAX = 500
    if len(y_val) > MAX:
        idx = np.random.choice(len(y_val),MAX, replace=False)
    else:
        idx = np.arange(len(y_val))
    plt.figure(figsize=(10,10))
    for i in range(6):
        x0 = np.min(y_val[idx,i])
        x1 = np.max(y_val[idx,i])
        plt.subplot(3,2,i+1)
        plt.scatter(y_val[idx,i],y_pred[idx,i])
        plt.xlabel('True '+labels[i])
        plt.ylabel('Predicted '+labels[i])
        plt.plot([x0,x1],[x0,x1],color='red')
        plt.axis('square')
    plt.savefig(plot_title+'.pdf')
    plt.show()
    
    
plot_y_yhat(np.array(outputVal), outputValPrediction, plot_title="y_yhat_val")
plot_y_yhat(np.array(outputTrain), outputTrainPrediction, plot_title="y_yhat_train")
