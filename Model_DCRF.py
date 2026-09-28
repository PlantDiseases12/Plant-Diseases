from sklearn.svm import SVC
import numpy as np

from Evaluation import evaluation


def Model_DCRF(train_data, train_target, test_data, test_target, Activation_Function, sol=None):
    sol1 = 1
    Kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    if sol1 == 1:
        clf = SVC(kernel=Kernels[sol1], degree=8, cache_size=int(sol[0]))
    else:
        clf = SVC(kernel=Kernels[sol1])

    pred = np.zeros(test_target.shape)
    # fitting x samples and y classes
    for i in range(test_target.shape[1]):
        clf.fit(train_data.tolist(), train_target[:, i].tolist())

        Y_pred = clf.predict(test_data.tolist())
        pred[:, i] = np.asarray(Y_pred)

    Eval = evaluation(pred, test_target)
    return Eval, pred