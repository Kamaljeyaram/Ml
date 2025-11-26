from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns, matplotlib.pyplot as plt
import pandas as pd

cancer_data = load_breast_cancer()
X = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
y = cancer_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB().fit(X_train, y_train)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred) * 100

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Naive Bayes - Confusion Matrix\nAccuracy = {acc:.2f}%")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.show()
