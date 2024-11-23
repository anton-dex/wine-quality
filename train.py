import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv('wine+quality/winequality-white.csv', sep=';')
df.columns = df.columns.str.replace(' ', '-').str.lower()
del df['residual-sugar']

df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 42)
df_train = df_full_train

df_train = df_train.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)

y_train = df_train.quality.values
y_test = df_test.quality.values

del df_train['quality']
del df_test['quality']

X_train = df_train.values
X_test = df_test.values

model_rf = RandomForestRegressor(n_estimators=50, max_depth=16, min_samples_leaf=4, random_state=42)
model_rf.fit(X_train, y_train)
y_test_pred = model_rf.predict(X_test)
print(round(root_mean_squared_error(y_test, y_test_pred), 3))

with open('model.pkl', 'wb') as f:
    pickle.dump(model_rf, f)
