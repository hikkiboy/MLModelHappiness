import pandas as pd 
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

file = './data/2019.csv'

happyData = pd.read_csv(file)

features = ['GDP per capita','Social support','Healthy life expectancy','Generosity']

X = happyData[features]
y = happyData.Overall_rank

X_train, X_valid, y_train, y_valid = train_test_split(X,y)

happyModel = XGBRegressor(n_estimators = 450, learning_rate = 0.07)
happyModel.fit(X_train, y_train,
               eval_set = [(X_valid, y_valid)],
               early_stopping_rounds = 5,
               verbose = False)

predictions = happyModel.predict(X_valid)

MAE = mean_absolute_error(predictions, y_valid)


print('Resultados reais:\n ')
print(y_valid.head())
print('Previsões do modelo')
print(predictions)
print("MAE:")
print(MAE)