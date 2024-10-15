import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

#data to use from the X_train.csv file
data = pd.read_csv('./X_train.csv')

# data clean up, removing the big amount of uneccessary 0.0
data_clean = data[(data['x_1'] != 0.0) | (data['y_1'] != 0.0) |
            (data['x_2'] != 0.0) | (data['y_2'] != 0.0) |
            (data['x_3'] != 0.0) | (data['y_3'] != 0.0)]

output_path = ('./data.csv')

# initial positions -> t=0.0
initial_positions = data_clean[(data_clean['t'] == 0.0)]
# group by positions to have unique initial positions
unique_initial_positions = initial_positions.drop_duplicates(subset=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])

comecaIndex = np.hstack((data_clean[data_clean.t == 0.0].index.values))
acabaIndex = np.hstack((data_clean[data_clean.t == 0.0].index.values - 1, data_clean.iloc[-1]['Id']))


data_clean.loc[:, ['x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position']] = 0.0
for j in range(comecaIndex.size):
    data_clean.loc[comecaIndex[j]:acabaIndex[j+1], ['x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position']] = data.loc[comecaIndex[j], ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values 

print(f"data_clean set size: {len(data_clean)}")

#data_clean = data_clean.drop(columns=['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3'], inplace=False)
# a prof diz que t > 0.0
data_clean = data_clean[(data_clean['t'] != 0.0)]
coluna_Id = data_clean.pop('Id')
data_clean.insert(0, 'Id', coluna_Id)

# create features
# relative positions to be used to create distances
data_clean['rel_x12'] = data_clean['x1_initial_position'] - data_clean['x2_initial_position']
data_clean['rel_y12'] = data_clean['y1_initial_position'] - data_clean['y2_initial_position']
data_clean['rel_x13'] = data_clean['x1_initial_position'] - data_clean['x3_initial_position']
data_clean['rel_y13'] = data_clean['y1_initial_position'] - data_clean['y3_initial_position']
data_clean['rel_x32'] = data_clean['x3_initial_position'] - data_clean['x2_initial_position']
data_clean['rel_y32'] = data_clean['y3_initial_position'] - data_clean['y2_initial_position']

# distances features
data_clean['d_12'] = np.sqrt(np.power(data_clean['rel_x12'], 2) + np.power(data_clean['rel_y12'], 2))
data_clean['d_13'] = np.sqrt(np.power(data_clean['rel_x13'], 2) + np.power(data_clean['rel_y13'], 2))
data_clean['d_32'] = np.sqrt(np.power(data_clean['rel_x32'], 2) + np.power(data_clean['rel_y32'], 2))

# velocities
data_clean['v_rel_x12'] = data_clean['v_x_1'] - data_clean['v_x_2']
data_clean['v_rel_y12'] = data_clean['v_y_1'] - data_clean['v_y_2']
data_clean['v_rel_x13'] = data_clean['v_x_1'] - data_clean['v_x_3']
data_clean['v_rel_y13'] = data_clean['v_y_1'] - data_clean['v_y_3']
data_clean['v_rel_x32'] = data_clean['v_x_3'] - data_clean['v_x_2']
data_clean['v_rel_y32'] = data_clean['v_y_3'] - data_clean['v_y_2']

# speeds
data_clean['speed_1'] = np.sqrt(np.power(data_clean['v_x_1'], 2) + np.power(data_clean['v_y_1'], 2))
data_clean['speed_2'] = np.sqrt(np.power(data_clean['v_x_2'], 2) + np.power(data_clean['v_y_2'], 2))
data_clean['speed_3'] = np.sqrt(np.power(data_clean['v_x_3'], 2) + np.power(data_clean['v_y_3'], 2))

# normalized velocities
data_clean['v_x_ratio_12'] = data_clean['v_x_1'] / data_clean['v_x_2']
data_clean['v_y_ratio_12'] = data_clean['v_y_1'] / data_clean['v_y_2']
data_clean['v_x_ratio_13'] = data_clean['v_x_1'] / data_clean['v_x_3']
data_clean['v_y_ratio_13'] = data_clean['v_y_1'] / data_clean['v_y_3']
data_clean['v_x_ratio_32'] = data_clean['v_x_3'] / data_clean['v_x_2']
data_clean['v_y_ratio_32'] = data_clean['v_y_3'] / data_clean['v_y_2']

# inverse of distance
data_clean['inv_d_12'] = 1 / data_clean['d_12']
data_clean['inv_d_13'] = 1 / data_clean['d_13']
data_clean['inv_d_32'] = 1 / data_clean['d_32']

# 80% to train, 20% to val
train, val = train_test_split(data_clean, test_size=0.2, random_state=42)
one_percent_train, _ = train_test_split(train, train_size=0.01, random_state=42)
one_percent_val, _ = train_test_split(val, train_size=0.01, random_state=42)

test = pd.read_csv('./X_test.csv')
cols = ['Id','t','x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position']
test.columns = cols

# Definindo as features de entrada e as variáveis alvo
features_X = [
    'x1_initial_position', 'y1_initial_position', 'x2_initial_position',
    'y2_initial_position', 'x3_initial_position', 'y3_initial_position',
    'rel_x12', 'rel_y12', 'rel_x13', 'rel_y13', 'rel_x32', 'rel_y32',
    'd_12', 'd_13', 'd_32', 'inv_d_12', 'inv_d_13', 'inv_d_32',
    'v_rel_x12', 'v_rel_y12', 'v_rel_x13', 'v_rel_y13', 'v_rel_x32', 'v_rel_y32',
    'speed_1', 'speed_2', 'speed_3',
    'v_x_ratio_12', 'v_y_ratio_12', 'v_x_ratio_13', 'v_y_ratio_13', 'v_x_ratio_32', 'v_y_ratio_32',
    't']

y = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

#sns.pairplot(data_clean[features_X + y].sample(200), kind="hist")
#plt.show()

#corr = data_clean[features_X + y].corr()
#sns.heatmap(corr, annot=True)
#sorted = corr.abs().unstack().sort_values(ascending=False)
#print(sorted.to_string())

#correlated_features = [
#    ('y1_initial_position',  'x1_initial_position'),
#    ('x1_initial_position',  'y1_initial_position'),
#    ('y2_initial_position',  'y3_initial_position'),
#    ('x3_initial_position',  'x2_initial_position')
#]

#features_to_remove = []
#for feature_pair in correlated_features:
#    features_to_remove.append(feature_pair[1])
    
#features_X = [f for f in features_X if f not in features_to_remove]

entry_train = train[features_X]
output_train = train[y]

entry_test  = test[features_X]

entry_val = val[features_X]
output_val = val[y]

#Pipeline pedida pela prof com standardscaler (padronização dos dados) e linear regression
pipeline = Pipeline([('scaler', StandardScaler()),
             ('linear_regression', LinearRegression())])

pipeline.fit(entry_train, output_train)

output_train_prediction = pipeline.predict(entry_train)
output_val_prediction = pipeline.predict(entry_val)
output_test_prediction = pipeline.predict(entry_test)

columns = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']
# Create a DataFrame from the predicted values
df_output = pd.DataFrame(output_test_prediction, columns=columns)
# Add a column for the 'id' values
df_output['id'] = df_output.index
# Reorder the columns to match the desired format
df_output = df_output[['id', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]
output_path = ('./augmented-baseline-model.csv')
df_output.to_csv(output_path,index=False)

rmse_train =  math.sqrt(mean_squared_error(output_train, output_train_prediction))
rmse_val = math.sqrt(mean_squared_error(output_val, output_val_prediction))

print(f"Train RMSE: {rmse_train}")
print(f"Validation RMSE: {rmse_val}")

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
    

#plot_y_yhat(np.array(output_val), output_val_prediction, plot_title="y_yhat_val")
#plot_y_yhat(np.array(output_train), output_train_prediction, plot_title="y_yhat_test")
