import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def plot_results(models, titles, X, y, plot_sv=False):
    # Set-up 2x2 grid for plotting.
    fig, sub = plt.subplots(1, len(titles))  # 1, len(list(models)))

    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    if len(titles) == 1:
        sub = [sub]
    else:
        sub = sub.flatten()
    for clf, title, ax in zip(models, titles, sub):
        # print(title)
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        if plot_sv:
            sv = clf.support_vectors_
            ax.scatter(sv[:, 0], sv[:, 1], c='k', s=60)

        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    plt.show()

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

C_hard = 1000000.0  # SVM regularization parameter
C = 10
n = 100



def get_circle_data(n=100, perturb=False):
    radius = np.hstack([np.random.random(n), np.random.random(n) + 1.5])
    angles = 2 * math.pi * np.random.random(2 * n)
    X1 = (radius * np.cos(angles)).reshape((2 * n, 1))
    X2 = (radius * np.sin(angles)).reshape((2 * n, 1))
    X = np.concatenate([X1, X2], axis=1)
    y = np.concatenate([np.ones((n, 1)), -np.ones((n, 1))], axis=0).reshape([-1])
    
    if perturb:
        for i in range(len(y)):
            if y[i] == -1 and np.random.random() < 0.1:
                y[i] = 1
    return X, y

def run_task_a(X, y):
    C = 10
    models = [
        svm.SVC(C=C, kernel='linear').fit(X, y),
        svm.SVC(C=C, kernel='poly', degree=2, coef0=0).fit(X, y),
        svm.SVC(C=C, kernel='poly', degree=3, coef0=0).fit(X, y) 
    ]
    titles = ['Linear', 'Homog. Poly Deg 2', 'Homog. Poly Deg 3']
    plot_results(models, titles, X, y)

def run_task_b(X, y):
    C = 10
    models = [
        svm.SVC(C=C, kernel='linear').fit(X, y),
        svm.SVC(C=C, kernel='poly', degree=2, coef0=1).fit(X, y),
        svm.SVC(C=C, kernel='poly', degree=3, coef0=1).fit(X, y) 
    ]
    titles = ['Linear', 'Non-Homog. Poly Deg 2', 'Non-Homog. Poly Deg 3']
    plot_results(models, titles, X, y)

def run_task_c():
    X_noisy, y_noisy = get_circle_data(perturb=True)
    C = 10
    models = [
        svm.SVC(C=C, kernel='poly', degree=2, coef0=1).fit(X_noisy, y_noisy),
        svm.SVC(C=C, kernel='rbf', gamma=10).fit(X_noisy, y_noisy)           
    ]
    titles = ['Poly Deg 2 (Noisy)', 'RBF gamma=10 (Noisy)']
    plot_results(models, titles, X_noisy, y_noisy)




if __name__ == "__main__":
    # Generate original clean data
    X, y = get_circle_data()
    
    # Execute separated task functions
    run_task_a(X, y)
    run_task_b(X, y)
    run_task_c()


