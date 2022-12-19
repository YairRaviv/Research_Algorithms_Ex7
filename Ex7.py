import random
from timeit import default_timer as timer
import numpy as np
import cvxpy
import matplotlib.pyplot as plt
import networkx as nx


# ====================================================== Question 1 ====================================================

def create_random_system(dim):
    solutions = [0] * dim
    for i in range(dim):
        solutions[i] = random.randint(1, 10000)
    mat = []
    for i in range(dim):
        mat.append([])
    for i in range(dim):
        for j in range(dim):
            mat[i].append(random.randint(1, 1000))
    return mat, solutions


def solve_numpy(systems):
    times = []
    for sys in systems:
        eq = np.array(sys[0])
        so = np.array(sys[1])
        d = len(sys[1])
        start = timer()
        val = np.linalg.solve(eq, so)
        end = timer()
        times.append((d, (end - start)*10000))
    return times


def solve_cvxpy(systems):
    times = []
    for sys in systems:
        d = len(sys[1])
        vars = cvxpy.Variable(d)
        eq = np.array(sys[0])
        so = np.array(sys[1])
        constrains = []
        sum = 0
        for i in range(d):
            for j in range(d):
                sum+= eq[i][j]*vars[j]
            constrains.append(sum == so[i])
        start = timer()
        prob = cvxpy.Problem(cvxpy.Minimize(1),constrains)
        end = timer()
        times.append((d, (end - start)*10000))
    return times



# ====================================================== Question 2 ====================================================


def cal_ratio(n):
    p = random.random()
    G = nx.gnp_random_graph(n,p)
    all_cliques = nx.find_cliques(G)
    max_size = 0
    for Q in all_cliques:
        if len(Q) > max_size:
            max_size = len(Q)
    app_size = len(nx.algorithms.approximation.max_clique(G))
    return app_size/max_size


# ====================================================== Question 3 ====================================================
# link to the solution - https://www.codingame.com/ide/puzzle/the-lucky-number






if __name__ == '__main__':
    systems = []
    for i in range(10,100):
        dim = i
        systems.append(create_random_system(dim))
    times_numpy = solve_numpy(systems)
    times_cvxpy = solve_cvxpy(systems)


    dims_nmpy = np.array(range(10,100))
    times_nmpy = np.array([x[1] for x in times_numpy])
    plt.plot(dims_nmpy,times_nmpy)

    dims_cvx = np.array(range(10, 100))
    times_cvx = np.array([x[1] for x in times_cvxpy])
    plt.plot(dims_cvx, times_cvx)
    plt.xlabel("Dimensions")
    plt.ylabel("Runtime")
    plt.show()

    ratios = []
    for n in range(30,70):
        ratios.append(cal_ratio(n))
    sizes = np.array(range(30,70))
    results = np.array(ratios)
    plt.plot(sizes, results)
    plt.xlabel("graph size")
    plt.ylabel("approximation ratio")
    plt.show()
