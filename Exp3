import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from sklearn.metrics import confusion_matrix

# ---- Load dataset from ONE working link ----
df = pd.read_csv("https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv")

# ---- Discretize important features ----
df['age'] = pd.cut(df['age'], [20,40,55,90], labels=["Young","Middle","Old"])
df['chol'] = pd.cut(df['chol'], [100,200,300,600], labels=["Low","Med","High"])
df['thalach'] = pd.cut(df['thalach'], [70,120,160,220], labels=["Low","Med","High"])
df['target'] = df['target'].map({0:"No",1:"Yes"})

# ---- Build Bayesian Network Model ----
model = DiscreteBayesianNetwork([('age','target'), ('chol','target'), ('thalach','target')])
model.fit(df[['age','chol','thalach','target']])
infer = VariableElimination(model)

# ---- Predict entire dataset to form CM ----
pred, true = [], []
for _, row in df.iterrows():
    q = infer.query(['target'], {'age':row['age'],'chol':row['chol'],'thalach':row['thalach']})
    pred.append(q.values.argmax())   # 0 or 1
    true.append(1 if row['target']=="Yes" else 0)

# ---- Print Confusion Matrix ----
print("\nConfusion Matrix:\n", confusion_matrix(true, pred))

# ---- Example Prediction ----
print("\nPrediction for (Old, High Cholesterol, Low Heart Rate):")
print(infer.query(['target'], {'age':'Old','chol':'High','thalach':'Low'}))
