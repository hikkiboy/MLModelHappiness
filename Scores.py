import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor


from sklearn.preprocessing import OneHotEncoder


file_path = './data/2019.csv'

spotify = pd.read_csv(file_path)

features = ['GDP per capita','Social support','Healthy life expectancy','Generosity']
X = spotify[features]
y = spotify.Overall_rank

# #função pra definir o score de determinado modelo baseado no learning rate
# def get_score(learning_rate):
#      pipes = Pipeline(steps = [
#          ('model', XGBRegressor(learning_rate = learning_rate))
#      ])
#      score = -1 * cross_val_score(pipes,X,y,cv = 3, scoring = 'neg_mean_absolute_error', error_score='raise')
#      return score.mean()

# #loop pra testar varios estimators diferentes
# results = {}
# for i in range (1,10):
#     results[0.01*i] = get_score(0.01*i)

# o melhor LR é 0.5
    


#função pra definir o score de determinado modelo baseado no numero de n_estimators
def get_score(n_estimators):
     pipes = Pipeline(steps = [
         ('model', XGBRegressor(n_estimators = n_estimators, learning_rate = 0.07))
     ])
     score = -1 * cross_val_score(pipes,X,y,cv = 3, scoring = 'neg_mean_absolute_error', error_score='raise')
     return score.mean()

#loop pra testar varios estimators diferentes
results = {}
for i in range (1,10):
    results[50*i] = get_score(50*i)



plt.plot(list(results.keys()), list(results.values()))
plt.show()
