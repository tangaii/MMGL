import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def modality_probe_accuracy(train_rep, test_rep, random_state):
    """Fit only on inner-training subjects, evaluate on held-out test subjects."""
    n_train, modal_num, dim = train_rep.shape
    n_test = test_rep.shape[0]
    if test_rep.shape[1:] != (modal_num, dim):
        raise ValueError("Probe training/test representation shapes differ")
    x_train = train_rep.reshape(n_train * modal_num, dim)
    x_test = test_rep.reshape(n_test * modal_num, dim)
    y_train = np.tile(np.arange(modal_num), n_train)
    y_test = np.tile(np.arange(modal_num), n_test)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", random_state=random_state)
    clf.fit(x_train, y_train)
    return float(accuracy_score(y_test, clf.predict(x_test)))
