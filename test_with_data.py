import pandas as pd
import pickle
import feature_engineering
import numpy as np

with open('../gve-manager/201707011237_projects.p', 'rb') as f:
    df = pickle.load(f)
with open('../gve-manager/time-to-fill.p', 'rb') as f:
    time = pickle.load(f)

df = df.merge(time, on='ProjectID', how='left')

df['SlowFlag'] = (df['DaysToFill'] > 2)*1
df['SlowFlag'][:3] = [1, 1, 1]

X = df[['Reason', 'RiskClass', 'G-PD', 'CreditSafe', 'Amount', 'Type', 'Interest', 'Maturity']]
y = df['SlowFlag']

engine = feature_engineering.Engine()
# X = engine.fit(X, y)
X = engine.genetic(X, y)
print(X.columns)
print(engine.score(X, y))
