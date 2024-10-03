import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

#data to use from the X_train.csv file
data = pd.read_csv(r'C:\Users\duart\Desktop\machine-learning-nova-2024-the-three-body-proble\mlNOVA\mlNOVA\X_train.csv')

#Colisions..identified
#collisions = data[(data['x_1'] == 0.0) & (data['y_1'] == 0.0) &
#                  (data['x_2'] == 0.0) & (data['y_2'] == 0.0) &
#                  (data['x_3'] == 0.0) & (data['y_3'] == 0.0)]
#print(f"Number of collision detected: {len(collisions)}")

print(f"data set size: {len(data)}")
# data clean up, removing the big amount of uneccessary 0.0
data_clean = data[(data['x_1'] != 0.0) | (data['y_1'] != 0.0) |
            (data['x_2'] != 0.0) | (data['y_2'] != 0.0) |
            (data['x_3'] != 0.0) | (data['y_3'] != 0.0)]
print(f"data_clean set size: {len(data_clean)}")

# initial positions -> t=0.0
initial_positions = data_clean[(data_clean['t'] == 0.0)]
print(f"initial_positions set size: {len(initial_positions)}")

# group by positions to have unique initial positions
unique_initial_positions = initial_positions.drop_duplicates(subset=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])
print(unique_initial_positions)
print("-------------------------")


stack = np.hstack((data_clean[data_clean.t == 0.0].index.values, 1284999))
print(stack)

new_data = data_clean.copy()
for i in range(stack.size):
    print(f"i = {i}")
    for j in range(stack[i], stack[i+1]):
        print(f"j = {j}")
        specific_row = unique_initial_positions.iloc[i]
        for col in unique_initial_positions.columns[1:]:
            new_data[col] = unique_initial_positions[col]




output_path = r'C:\Users\duart\Desktop\machine-learning-nova-2024-the-three-body-proble\mlNOVA\mlNOVA\new_data.csv'
new_data.to_csv(output_path,index=False)








#80% to train, 10% to test, 10% to val... we can try 70% 15% 15% aswell
train, test_val = train_test_split(unique_initial_positions, test_size=0.3, random_state=42)
test, val = train_test_split(test_val, test_size=0.5, random_state=42)

print(f"Train set size: {len(train)}")
print(f"Validation set size: {len(val)}")
print(f"Test set size: {len(test)}")
print(f"Total before clean =  1285002")



"""
#Function do clean the data (aka remove de colisions)
def cleanData(dataToClean, data):
    dataToClean_data = []
    for ids in dataToClean.index:
        next_indices = range(ids, 257 + ids)
        dataToClean_data.append(data.iloc[next_indices])
    
    dataToCleanAsDataStream = pd.concat(dataToClean_data, ignore_index=True)
    cleanData = dataToCleanAsDataStream[
        (dataToCleanAsDataStream['x_1'] != 0.0) &
        (dataToCleanAsDataStream['y_1'] != 0.0) &
        (dataToCleanAsDataStream['x_2'] != 0.0) &
        (dataToCleanAsDataStream['y_2'] != 0.0) &
        (dataToCleanAsDataStream['x_3'] != 0.0) &
        (dataToCleanAsDataStream['y_3'] != 0.0)
    ] 
    return cleanData

#final train, test and val data.
train_data = cleanData(train, data)
test_data = cleanData(test, data)
val_data = cleanData(val, data)
print(len(train_data))
print(len(test_data))
print(len(val_data))
########################### Task 1.1 Ends Here#################################

#Regression é um função que vai prever os valores com base em outros casos analisados
#inputs são regressors/predictors e outputs são responses 

#dados de entrada
entryLabels = ['t', 'v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3']

#o que queremos prever 
outputLabels =  ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

entry_train = train_data[entryLabels]
output_train = train_data[outputLabels]
entry_test  = test_data[entryLabels]
output_test = test_data[outputLabels]
entry_val = val_data[entryLabels]
output_val = val_data[outputLabels]

#Pipeline pedida pela prof com standardscaler (padronização dos dados) e linear regression
pipeline = Pipeline([('scaler', StandardScaler()),
             ('linear_regression', LinearRegression())])

pipeline.fit(entry_train, output_train)

output_train_prediction = pipeline.predict(entry_train)
output_val_prediction = pipeline.predict(entry_val)
output_test_prediction = pipeline.predict(entry_test)

mse_train = mean_squared_error(output_train, output_train_prediction)
mse_val = mean_squared_error(output_val, output_val_prediction)
mse_test = mean_squared_error(output_test, output_test_prediction)

#se o MSE do dado de treino for muito menor que o MSE de validação e de treino --> modelo está a sofrer overfitting (modelo ajusta-se demasiado aos dados de treino)
print(f"Train MSE: {mse_train}")
print(f"Validation MSE: {mse_val}")
print(f"Test MSE: {mse_test}")

def plot_y_yhat(y_test,y_pred, plot_title = "plot"):
    labels = ['x_1','y_1','x_2','y_2','x_3','y_3']
    MAX = 500
    if len(y_test) > MAX:
        idx = np.random.choice(len(y_test),MAX, replace=False)
    else:
        idx = np.arange(len(y_test))
    plt.figure(figsize=(10,10))
    for i in range(6):
        x0 = np.min(y_test[idx,i])
        x1 = np.max(y_test[idx,i])
        plt.subplot(3,2,i+1)
        plt.scatter(y_test[idx,i],y_pred[idx,i])
        plt.xlabel('True '+labels[i])
        plt.ylabel('Predicted '+labels[i])
        plt.plot([x0,x1],[x0,x1],color='red')
        plt.axis('square')
    plt.savefig(plot_title+'.pdf')
    plt.show()
    
    
plot_y_yhat(np.array(output_val), output_val_prediction, plot_title="y_yhat_val")
plot_y_yhat(np.array(output_test), output_test_prediction, plot_title="y_yhat_test")
"""