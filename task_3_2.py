import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV

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

# 80% to train, 20% to val, use only 1% of the data 
train, val = train_test_split(dataClean, test_size=0.2, random_state=42)
onePercentTrain, _ = train_test_split(train, train_size=0.01, random_state=42)
onePercentVal, _ = train_test_split(val, train_size=0.01, random_state=42)

# get the X_test and change the name of the columns
test = pd.read_csv('./X_test.csv')
cols = ['Id','t','x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position']
test.columns = cols

# features are the columns of entrance, to get the final positions, depending on the time and initial positions (that are the features)
featuresX = ['x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position', 't']
# y are the final positions, calculated depending on the initial positions and time
y = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

# y1 and x1 are not displaying any values, which means that both can be eliminated since it will not afect the model
# y2 and y3 are highly correlated between each other, one needs to be eliminated
# x3 and x2 are highly correlated between each other, one needs to be eliminated
correlatedFeatures = [
    ('y1_initial_position',  'x1_initial_position'), 
    ('x1_initial_position',  'y1_initial_position'),
    ('y2_initial_position',  'y3_initial_position'),
    ('x3_initial_position',  'x2_initial_position')
]

# eliminating the second element of the pair of features
# no need to eliminate both, because if one it's eliminated then the other one will not be correlated with nothing else
featuresToRemove = []
for featurePair in correlatedFeatures:
    featuresToRemove.append(featurePair[1])
    
featuresX = [feature for feature in featuresX if feature not in featuresToRemove]

# assigning the inputs and outputs
entryTrain = onePercentTrain[featuresX]
outputTrain = onePercentTrain[y]
entryVal = onePercentVal[featuresX]
outputVal = onePercentVal[y]

entryTest = test[featuresX]

# degree chosen was 14, no need to do loops
def validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None, degree=14, max_features=None):
    bestRmse = 2
    bestModel = None
    bestDegree = None
    bestAlpha = None

    # loop degrees
    #for degree in degrees:
    regressor = RidgeCV(alphas=np.logspace(-6, 6, 13), store_cv_results=True)
        # pipeline with StandardScaler and PolynomialFeatures
    polyRegModel = make_pipeline(PolynomialFeatures(degree),  StandardScaler(), regressor)
    polyRegModel.fit(X_train, y_train)
    
    yPred = polyRegModel.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, yPred))

    polyFeatures = polyRegModel.named_steps['polynomialfeatures']
    print(f"Degree {degree}: Number of features = {polyFeatures.n_output_features_}, Best alpha: {regressor.alpha_}")

        # updates to the best parameters of the model
    if rmse < bestRmse:
        bestRmse = rmse
        bestModel = polyRegModel
        bestDegree = degree            
        bestAlpha = regressor.alpha_

    print(f"Best Degree: {bestDegree}, Best Alpha: {bestAlpha}, Best RMSE: {bestRmse}")
    return bestModel, bestRmse, bestDegree, bestAlpha

bestModel, bestRmse, bestDegree, bestAlpha = validate_poly_regression(entryTrain, outputTrain, entryVal, outputVal)

outputValPrediction = bestModel.predict(entryVal)
outputTrainPrediction = bestModel.predict(entryTrain)

# calculate rmse values 
rmseTrain = math.sqrt(mean_squared_error(outputTrain, outputTrainPrediction))
rmseVal = math.sqrt(mean_squared_error(outputVal, outputValPrediction))

print(f"Train RMSE (best model): {rmseTrain}")
print(f"Validation RMSE (best model): {rmseVal}")

# plot
def plot_y_yhat(y_val, y_pred, plot_title="plot"):
    labels = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
    MAX = 500
    if len(y_val) > MAX:
        idx = np.random.choice(len(y_val), MAX, replace=False)
    else:
        idx = np.arange(len(y_val))
    plt.figure(figsize=(10, 10))
    for i in range(6):
        x0 = np.min(y_val[idx, i])
        x1 = np.max(y_val[idx, i])
        plt.subplot(3, 2, i+1)
        plt.scatter(y_val[idx, i], y_pred[idx, i])
        plt.xlabel('True ' + labels[i])
        plt.ylabel('Predicted ' + labels[i])
        plt.plot([x0, x1], [x0, x1], color='red')
        plt.axis('square')
    plt.savefig(plot_title + '.pdf')
    plt.show()

plot_y_yhat(np.array(outputVal), outputValPrediction, plot_title="y_yhat_val_polynomial")
plot_y_yhat(np.array(outputTrain), outputTrainPrediction, plot_title="y_yhat_train_polynomial")

columns = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
# create a DataFrame from the predicted values
df = pd.DataFrame(bestModel.predict(entryTest), columns=columns)
# add a column for the 'id' and reorder the columns to have the output file (submission file) in the expected structure
df['id'] = df.index
df = df[['id', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]

# extract the submission file
outputPath = ('./reduced_polynomial_submission.csv')
df.to_csv(outputPath,index=False)
