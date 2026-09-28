import numpy as np
import time


def AZO(X, fitness, lowerbound, upperbound, Max_iterations):
    [SearchAgents, dimension] = X.shape

    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        L = X[i, :]
        fit[i] = fitness(L)

    Xbest = X[np.argmin(fit), :]
    fbest = np.min(fit)

    SOA_curve = np.zeros(Max_iterations)
    ct = time.time()

    for t in range(Max_iterations):
        for i in range(SearchAgents):
            K = np.where(fit < fit[i])[0]

            X_P1 = np.maximum(X, lowerbound)
            X_P1 = np.minimum(X_P1, upperbound)

            L = X_P1
            F_P1 = fitness(L)

            if F_P1[i] < fit[i]:
                X[i, :] = X_P1[i]
                fit[i] = F_P1[i]

        for i in range(SearchAgents):
            if np.random.rand() < 0.5:
                X_P2 = X[i, :] + (1 - 2 * np.random.rand(dimension)) / (t + 1) * X[i, :]
                X_P2 = np.maximum(X_P2, lowerbound)
                X_P2 = np.minimum(X_P2, upperbound)
            else:
                X_P2 = X[i, :] + lowerbound / (t + 1) + np.random.rand(1) * (upperbound / (t + 1) - lowerbound / (t + 1))
                X_P2 = np.maximum(X_P2, lowerbound / (t + 1))
                X_P2 = np.minimum(X_P2, upperbound)
                X_P2 = np.maximum(X_P2, lowerbound)
                X_P2 = np.minimum(X_P2, upperbound)

            L = X_P2
            F_P2 = fitness(L)

            if F_P2[i] < fit[i]:
                X[i, :] = X_P2[i]
                fit[i] = F_P2[i]

        Best_score = fbest
        Best_pos = Xbest
        SOA_curve[t] = fbest
    ct -= time.time()

    return Best_score, SOA_curve, Best_pos, ct
