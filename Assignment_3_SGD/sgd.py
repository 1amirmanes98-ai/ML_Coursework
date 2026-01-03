#################################
# Your name:
#################################


import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper():
    mnist = fetch_openml('mnist_784', as_frame=False)
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels



def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements SGD for hinge loss with L2-regularization.
    """
    n_samples, n_features = data.shape
    w = np.zeros(n_features) # Initialize w1 = 0 
    
    for t in range(1, T + 1):
        i = np.random.randint(0, n_samples) # Sample i uniformly 
        eta_t = eta_0 / t # eta_t = eta_0 / t 

        # Check condition: y_i * <w, x_i> < 1 
        if labels[i] * np.dot(w, data[i]) < 1:
            w = (1 - eta_t) * w + eta_t * C * labels[i] * data[i] # Update w_t+1 
        else:
            w = (1 - eta_t) * w # Update w_t+1
    return w




def run_assignment_tasks():
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = helper()
    
    # (a) Cross-validation for eta_0 [cite: 207]
    eta_range = [10**i for i in range(-5, 6)]
    eta_accuracies = []
    for eta in eta_range:
        accs = [np.mean(np.sign(np.dot(X_val, SGD_hinge(X_train, y_train, 1, eta, 1000))) == y_val) for _ in range(10)]
        eta_accuracies.append(np.mean(accs))
    
    best_eta = eta_range[np.argmax(eta_accuracies)]
    plt.semilogx(eta_range, eta_accuracies, marker='o')
    plt.title("Val Accuracy vs eta_0")
    plt.show()

    # (b) Cross-validation for C [cite: 210]
    C_range = [10**i for i in range(-5, 6)]
    C_accuracies = []
    for c_val in C_range:
        accs = [np.mean(np.sign(np.dot(X_val, SGD_hinge(X_train, y_train, c_val, best_eta, 1000))) == y_val) for _ in range(10)]
        C_accuracies.append(np.mean(accs))
        
    best_C = C_range[np.argmax(C_accuracies)]
    plt.semilogx(C_range, C_accuracies, marker='o')
    plt.title(f"Val Accuracy vs C (eta_0={best_eta})")
    plt.show()

    # (c) Visualization [cite: 212, 213]
    w_final = SGD_hinge(X_train, y_train, best_C, best_eta, 20000)
    plt.imshow(w_final.reshape(28, 28), interpolation='nearest')
    plt.title("Weight Visualization (T=20000)")
    plt.show()

    # (d) Test Accuracy [cite: 217]
    test_acc = np.mean(np.sign(np.dot(X_test, w_final)) == y_test)
    print(f"Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    run_assignment_tasks()

