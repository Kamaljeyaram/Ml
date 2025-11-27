import numpy as np
import matplotlib.pyplot as plt

# Sample nonlinear data
np.random.seed(0)
X = np.linspace(-3, 3, 40)
y = np.sin(X) + 0.2*np.random.randn(40)

# Add bias term
X_train = np.c_[np.ones_like(X), X]

# LWR function
def lwr(xq, X, y, tau=0.5):
    w = np.exp(-(X[:,1]-xq)**2/(2*tau*tau))
    W = np.diag(w)
    theta = np.linalg.pinv(X.T @ W @ X) @ X.T @ W @ y
    return np.array([1, xq]) @ theta

# Predict for multiple points
Xq = np.linspace(-3, 3, 200)
yp = [lwr(x, X_train, y) for x in Xq]

# Plot
plt.scatter(X, y, label="Data")
plt.plot(Xq, yp, "r", label="LWR Curve")
plt.plot(Xq, np.sin(Xq), "g--", label="True sin(x)")
plt.legend(); plt.title("Locally Weighted Regression"); plt.show()
