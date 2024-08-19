import math
import random
import numpy as np
import sympy as sp
from sympy import symbols

input_file = "./dataset/100_1.txt"
output_file = "./result/100_1.txt"

X = np.loadtxt(input_file, dtype=int)
N = len(X)

p_mutation = 0.2
neighbor_size = 3
barrier_length = 1000
# max_generation = 1000
max_generation = 0
k = 2
gamma = 0.5
beta = 1

lamb = []
z = [None, None]
z_nad = [None, None]
population = []
archive_f = []
archive_fitness = []
archive_r = []
archive_individual = []


def init_lambda():
    lamb = []
    for i in range(9):
        a = i * 3 + 1
        b = 10 - a
        if b <= 0:
            continue
        lamb.append([a / 10, b / 10])
    return lamb


def search_neighbor(lamb):
    neighbor = []
    distance = []
    num = len(lamb)
    for i in range(num):
        distance.append([0 for _ in range(num)])

    for i in range(num):
        for j in range(i, num):
            lamb1 = lamb[i]
            lamb2 = lamb[j]
            dis = 0
            for k in range(2):
                dis += (lamb1[k] - lamb2[k]) ** 2
            dis = dis**0.5
            distance[i][j] = dis
            distance[j][i] = dis
        temp_list = np.array(distance[i])
        index = np.argsort(temp_list)
        index = index.tolist()
        neighbor.append(index[:neighbor_size])
    return neighbor


def initIndividual():
    individual = []
    for _ in range(N):
        individual.append(random.randint(0, 1))
    return individual


def is_all_zero(array):
    for i in array:
        if i != 0:
            return False
    return True


def initValidIndividual():
    individual = initIndividual()
    while is_all_zero(individual):
        individual = initIndividual()
    return individual


def initPopulation(pop_size):
    population = []
    for _ in range(pop_size):
        individual = initValidIndividual()
        population.append(individual)
    return population


def mutation(individual):
    newIndividual = individual
    for _ in range(4):
        index = random.randint(0, N - 1)
        newIndividual[index] = (int(newIndividual[index]) + 1) % 2
    return newIndividual


def crossover(parent1, parent2):
    gap = 5
    child = []
    for i in range(0, N, gap):
        rand = random.random()
        if rand > 0.5:
            child += parent1[i : i + gap]
        else:
            child += parent2[i : i + gap]

    # mutation
    if random.random() < p_mutation:
        child = mutation(child)

    return child


# note
# r_s = k * r_u
# r_s - r_u = (k - 1) * r_u
def radius_formalize_outermost_sensor(x1, isFirst=True):
    if isFirst == False:
        x1 = barrier_length - x1

    r_u1 = symbols("r_u1", integer=True, positive=True)
    expr = sp.exp(-beta * (x1 - (k - 1) * r_u1)) - gamma
    sol = sp.solveset(expr >= 0, r_u1, domain=sp.S.Reals)
    if not sol:
        print("Không tìm thấy giá trị nhỏ nhất của r_u1 thoả mãn bất phương trình.")
        return None

    # print(f"Tập nghiệm của r_u1: {sol}")
    if isinstance(sol, sp.Interval):
        min_r_u1 = sol.inf
        print(f"Giá trị nhỏ nhất của r_u1: {math.ceil(min_r_u1)}")
        if min_r_u1 < 0:
            return 0
        return math.ceil(min_r_u1)
    elif isinstance(sol, sp.Union):
        min_r_u1 = min([s.inf for s in sol.args if isinstance(s, sp.Interval)])
        print(f"Giá trị nhỏ nhất của r_u1*: {math.ceil(min_r_u1)}")
        if min_r_u1 < 0:
            return 0
        return math.ceil(min_r_u1)
    else:
        print("Không tìm thấy giá trị nhỏ nhất của r_u1 thoả mãn bất phương trình.")
        return None


def radius_formalize_sensor(r_u1, x1, x2):
    if x1 + r_u1 * (k - 1) >= x2:
        return 0

    x = symbols("x", integer=True, positive=True)
    expr = sp.exp(-beta * (x - x1 - (k - 1) * r_u1)) - gamma
    sol = sp.solveset(expr >= 0, x, domain=sp.S.Reals)
    r_certain1 = sol.sup

    if x1 + r_certain1 >= x2:
        return 0

    r_certain1 = math.floor(r_certain1)

    r_u2_max = (x2 - x1 - r_certain1) / (k - 1)
    r_u2_min = (x2 - x1 - r_certain1) / k

    r_u2, x = symbols("r_u2 x", integer=True, positive=True)
    expr = (
        sp.exp(-beta * (x - x1 - (k - 1) * r_u1))
        + sp.exp(-beta * (x2 - x - (k - 1) * r_u2))
        - gamma
    )

    for r2_val in range(math.ceil(r_u2_min), math.ceil(r_u2_max) + 1):
        x_val = (x1 + x2 - (k - 1) * (r_u1 - r2_val)) / 2
        print("x_val: ", x_val)
        inequality = expr.subs({r_u2: r2_val, x: x_val})
        print(inequality)
        if inequality >= 0:
            return r2_val

    return 0


def radius_formalize(individual):
    # Ex: all_r_u = [0, 100, 0, 0, 59, 74, 0,]; r_u = [100, 59, 74]
    index = [i for i in range(N) if individual[i] == 1]

    r_u, all_r_u, r_0_count = [], [], 0
    r_u.append(radius_formalize_outermost_sensor(X[index[0]]))

    for i in range(1, len(index)):
        r_temp = radius_formalize_sensor(
            r_u[i - 1 - r_0_count], X[index[i - 1 - r_0_count]], X[index[i]]
        )
        if r_temp == 0:
            r_0_count += 1
        else:
            r_0_count = 0

        r_u.append(r_temp)

    r_last = radius_formalize_outermost_sensor(X[index[-1]], isFirst=False)
    if r_last > r_u[-1]:
        print("r_last = ", r_last)
        r_u[-1] = r_last

    r_index = 0
    for i in range(N):
        if individual[i] == 1:
            all_r_u.append(r_u[r_index])
            r_index += 1
        else:
            all_r_u.append(0)

    return all_r_u


def calculate_energy_consumption(r_u):
    total_energy_consumption = 0
    total_energy_consumption += 1 / 2 * ((k - 1) * r_u) ** 2
    +(beta * math.exp(-beta * r_u) * (1 + r_u)) / (beta**2)

    return total_energy_consumption


def evaluate(individual):
    all_r_u = radius_formalize(individual)
    total_active_sunsor = 0
    total_enegy_consumption = 0
    for i in range(N):
        if all_r_u[i] > 0:
            total_active_sunsor += 1
            total_enegy_consumption += calculate_energy_consumption(all_r_u[i])

    return total_active_sunsor, total_enegy_consumption, all_r_u


# chose min fitness
def calculate_fitness(f1, f2, z, z_nad, lamb_i):
    fitness = 0
    fitness += lamb_i[0] * (f1 - z[0]) / (z_nad[0] - z[0])
    fitness += lamb_i[1] * (f2 - z[1]) / (z_nad[1] - z[1])
    return fitness


def main():
    lamb = init_lambda()
    # lamb = [
    #     [0.1, 0.9],
    #     [0.2, 0.8],
    #     [0.3, 0.7],
    #     [0.4, 0.6],
    #     [0.5, 0.5],
    #     [0.6, 0.4],
    #     [0.7, 0.3],
    #     [0.8, 0.2],
    #     [0.9, 0.1],
    # ]

    pop_size = len(lamb)
    print("pop_size = ", pop_size)
    # pop_size = 2
    # pop_size = 9

    neighbor = search_neighbor(lamb)
    print("neighbor = ", neighbor)
    # neighbor = [
    #     [0, 1, 2, 3, 4],
    #     [1, 0, 2, 3, 4],
    #     [2, 1, 3, 4, 0],
    #     [3, 4, 2, 5, 1],
    #     [4, 3, 5, 2, 6],
    #     [5, 4, 6, 3, 7],
    #     [6, 5, 7, 4, 8],
    #     [7, 8, 6, 5, 4],
    #     [8, 7, 6, 5, 4],
    # ]

    population = initPopulation(pop_size)
    r = [None] * pop_size
    f1 = [None] * pop_size
    f2 = [None] * pop_size
    fitness = [None] * pop_size

    # init
    for i in range(pop_size):
        f1[i], f2[i], r[i] = evaluate(population[i])
        archive_individual.append(population[i])
        archive_r.append(r[i])
        archive_f.append([f1[i], f2[i]])

    z = [min(f1), min(f2)]
    z_nad = [max(f1), max(f2)]

    for i in range(pop_size):
        fitness[i] = calculate_fitness(f1[i], f2[i], z, z_nad, lamb[i])
        archive_fitness.append(fitness[i])

    for generation in range(max_generation):
        print("generation =", generation)
        for i in range(pop_size):
            print(i)
            individual = population[i]
            index1, index2 = random.sample(range(0, neighbor_size), 2)
            parent1, parent2 = (
                population[neighbor[i][index1]],
                population[neighbor[i][index2]],
            )

            child = crossover(parent1, parent2)

            total_active_sunsor, total_enegy_consumption, radii = evaluate(child)
            child_fit = calculate_fitness(
                total_active_sunsor, total_enegy_consumption, z, z_nad, lamb[i]
            )

            if child_fit < fitness[i]:
                population[i] = child
                fitness[i] = child_fit
                r[i] = radii
                f1[i] = total_active_sunsor
                f2[i] = total_enegy_consumption

            archive_r.append(r[i])
            archive_f.append([f1[i], f2[i]])
            archive_fitness.append(fitness[i])
            archive_individual.append(population[i])

        z = [min(f1), min(f2)]
        z_nad = [max(f1), max(f2)]

    for j in range(len(archive_r)):
        print("j = ", j)
        print("archive_individual = ", archive_individual[j])
        print("archive_fitness = ", archive_fitness[j])
        print("archive_r = ", archive_r[j])
        print("archive_f = ", archive_f[j])


if __name__ == "__main__":
    main()
