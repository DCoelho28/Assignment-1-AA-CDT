import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
from sklearn.feature_selection import SelectKBest, f_regression

# Data loading and cleaning remains unchanged
data = pd.read_csv('./X_train.csv')

# Data cleaning step
data_clean = data[(data['x_1'] != 0.0) | (data['y_1'] != 0.0) |
            (data['x_2'] != 0.0) | (data['y_2'] != 0.0) |
            (data['x_3'] != 0.0) | (data['y_3'] != 0.0)]

initial_positions = data_clean[(data_clean['t'] == 0.0)]
unique_initial_positions = initial_positions.drop_duplicates(subset=['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3'])

comecaIndex = np.hstack((data_clean[data_clean.t == 0.0].index.values))
acabaIndex = np.hstack((data_clean[data_clean.t == 0.0].index.values - 1, data_clean.iloc[-1]['Id']))

data_clean.loc[:, ['x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position']] = 0.0

for j in range(comecaIndex.size):
    data_clean.loc[comecaIndex[j]:acabaIndex[j+1], ['x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position']] = data.loc[comecaIndex[j], ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']].values 

data_clean = data_clean.drop(columns=['v_x_1', 'v_y_1', 'v_x_2', 'v_y_2', 'v_x_3', 'v_y_3'], inplace=False)
data_clean = data_clean[(data_clean['t'] != 0.0)]
coluna_Id = data_clean.pop('Id')
data_clean.insert(0, 'Id', coluna_Id)

# 80% to train, 20% to val
train, val = train_test_split(data_clean, test_size=0.2, random_state=42)
one_percent_train, _ = train_test_split(train, train_size=0.004, random_state=42)
one_percent_val, _ = train_test_split(val, train_size=0.004, random_state=42)

test = pd.read_csv('./X_test.csv')
cols = ['Id','t','x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position']
test.columns = cols

# Definindo as features de entrada e as variáveis alvo
features_X = ['x1_initial_position', 'y1_initial_position', 'x2_initial_position', 'y2_initial_position', 'x3_initial_position', 'y3_initial_position', 't']
y = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

#sns.pairplot(data_clean[features_X + y].sample(200), kind="hist")

corr = data_clean[features_X + y].corr()
#sns.heatmap(corr, annot=True)
sorted = corr.abs().unstack().sort_values(ascending=False)
print(sorted.to_string())

correlated_features = [
    ('y1_initial_position',  'x1_initial_position'),
    ('x1_initial_position',  'y1_initial_position'),
    ('y2_initial_position',  'y3_initial_position'),
    ('x3_initial_position',  'x2_initial_position')
]

features_to_remove = []
for feature_pair in correlated_features:
    features_to_remove.append(feature_pair[1])
    
features_X = [f for f in features_X if f not in features_to_remove]

# Usar apenas 1% dos dados de treino e validação
entry_train = one_percent_train[features_X]
output_train = one_percent_train[y]

entry_val = one_percent_val[features_X]
output_val = one_percent_val[y]

entry_test = test[features_X]

def validate_poly_regression(X_train, y_train, X_val, y_val, regressor=None, degrees=range(1,15), max_features=None):
    best_rmse = float('inf')
    best_model = None
    best_degree = None
    best_alpha = None

    # Loop pelos graus polinomiais
    for degree in degrees:
    # RidgeCV para ajustar os valores de alpha
        regressor = RidgeCV(alphas=np.logspace(-6, 6, 13), store_cv_results=True)
    #select = SelectKBest(f_regression, k=50)
    # Pipeline com as transformações necessárias
        poly_reg_model = make_pipeline(PolynomialFeatures(degree),  StandardScaler(), regressor)
        poly_reg_model.fit(X_train, y_train)
    # Previsões
        y_pred = poly_reg_model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    # Número de características geradas
        poly_features = poly_reg_model.named_steps['polynomialfeatures']
        print(f"Degree {degree}: Number of features = {poly_features.n_output_features_}, Best alpha: {regressor.alpha_}")

    # Atualiza se o RMSE for melhor
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = poly_reg_model
            best_degree = degree
            best_alpha = regressor.alpha_

    print(f"Best Degree: {best_degree}, Best Alpha: {best_alpha}, Best RMSE: {best_rmse}")
    return best_model, best_rmse, best_degree, best_alpha

# Rodando a validação
    
#for n in range(0,9):
best_model, best_rmse, best_degree, best_alpha = validate_poly_regression(entry_train, output_train, entry_val, output_val)
#    print(f"Best Model: {best_model},Best Degree: {best_degree}, Best Alpha: {best_alpha}, Best RMSE: {best_rmse}")


# Previsões com o melhor modelo
output_val_prediction = best_model.predict(entry_val)
output_train_prediction = best_model.predict(entry_train)
# output_test_prediction= best_model.predict(entry_test)
# Cálculo do RMSE no conjunto de treino e validação
rmse_train = math.sqrt(mean_squared_error(output_train, output_train_prediction))
rmse_val = math.sqrt(mean_squared_error(output_val, output_val_prediction))

print(f"Train RMSE (best model): {rmse_train}")
print(f"Validation RMSE (best model): {rmse_val}")

# Função para plotar os gráficos y vs y_pred
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

# Plotando os resultados
#plot_y_yhat(np.array(output_val), output_val_prediction, plot_title="y_yhat_val")
#plot_y_yhat(np.array(output_train), output_train_prediction, plot_title="y_yhat_train")

# Previsões finais para o conjunto de teste
columns = ['x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']

df_output = pd.DataFrame(best_model.predict(entry_test), columns=columns)
# df_output = pd.DataFrame(output_test_prediction, columns=columns)
df_output['id'] = df_output.index
df_output = df_output[['id', 'x_1', 'y_1', 'x_2', 'y_2', 'x_3', 'y_3']]

output_path = ('./polynomial_submission.csv')
df_output.to_csv(output_path,index=False)
