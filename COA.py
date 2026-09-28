import numpy as np
import time


# Crocodile Optimization Algorithm (COA)
def COA(initsol, fobj, xmin, xmax, Max_iter):
    N, dim = initsol.shape  # Number of crocodiles and dimensions

    # Initialize population
    crocs = initsol.copy()
    fitness = np.array([fobj(crocs[i]) for i in range(N)])

    # Initialize global best
    best_idx = np.argmin(fitness)
    gBest = crocs[best_idx].copy()
    gBestScore = fitness[best_idx]

    # Convergence curve
    Convergence_curve = np.zeros(Max_iter)
    Convergence_curve[0] = gBestScore

    start_time = time.time()

    for t in range(1, Max_iter):
        r = np.random.rand()  # random number for phase selection

        # --- Phase 1: Exploration (search for prey) ---
        for i in range(N):
            leader = gBest
            random_croc = crocs[np.random.randint(0, N)]
            new_position = crocs[i] + r * (random_croc - crocs[i]) + np.random.randn(dim) * 0.1
            new_position = np.clip(new_position, xmin, xmax)
            new_fitness = fobj(new_position)

            if new_fitness < fitness[i]:
                crocs[i] = new_position
                fitness[i] = new_fitness

        # --- Phase 2: Exploitation (ambush & attack) ---
        for i in range(N):
            attack_vector = gBest - crocs[i]
            new_position = crocs[i] + (1 - r) * attack_vector + np.random.randn(dim) * 0.05
            new_position = np.clip(new_position, xmin, xmax)
            new_fitness = fobj(new_position)

            if new_fitness < fitness[i]:
                crocs[i] = new_position
                fitness[i] = new_fitness

        # Update global best
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < gBestScore:
            gBest = crocs[best_idx].copy()
            gBestScore = fitness[best_idx]

        Convergence_curve[t] = gBestScore

    total_time = time.time() - start_time
    return gBest, gBestScore, Convergence_curve, total_time
