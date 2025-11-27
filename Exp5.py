from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Load iris dataset
iris = load_iris()
X, y = iris.data, iris.target
names = iris.target_names

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Print correct predictions
print("===== CORRECT PREDICTIONS =====")
for i in range(len(pred)):
    if pred[i] == y_test[i]:
        print(f"Predicted: {names[pred[i]]:<12} | Actual: {names[y_test[i]]}")

# Print wrong predictions
print("\n===== WRONG PREDICTIONS =====")
for i in range(len(pred)):
    if pred[i] != y_test[i]:
        print(f"Predicted: {names[pred[i]]:<12} | Actual: {names[y_test[i]]}")

# Print confusion matrix
cm = confusion_matrix(y_test, pred)
print("\n===== CONFUSION MATRIX =====")
print(cm)
