import math
import time
import random
import numpy as np
import sympy as sp
from sympy import symbols
import multiprocessing

input_file = "./dataset/100_1.txt"
output_file = "./result/100_1.txt"

X = np.loadtxt(input_file, dtype=int)
N = len(X)

p_mutation = 0.2
neighbor_size = 5
barrier_length = 1000
# max_generation = 1000
max_generation = 100
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
        a = i + 1
        b = 10 - a
        if b <= 0:
            continue
        lamb.append([a / 10, b / 10])
    return lamb


lamb = init_lambda()
pop_size = len(lamb)
r = [None] * pop_size
f1 = [None] * pop_size
f2 = [None] * pop_size
fitness = [None] * pop_size


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


neighbor = search_neighbor(lamb)


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


population = initPopulation(pop_size)


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
        # print(f"Giá trị nhỏ nhất của r_u1: {math.ceil(min_r_u1)}")
        if min_r_u1 < 0:
            return 0
        return math.ceil(min_r_u1)
    elif isinstance(sol, sp.Union):
        min_r_u1 = min([s.inf for s in sol.args if isinstance(s, sp.Interval)])
        # print(f"Giá trị nhỏ nhất của r_u1*: {math.ceil(min_r_u1)}")
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
    x1_certain = sol.sup

    if x1_certain >= x2:
        return 0

    x1_certain = math.floor(x1_certain)

    r_u2_max = (x2 - x1_certain) / (k - 1)
    r_u2_min = (x2 - x1_certain) / k

    r_u2, x = symbols("r_u2 x", integer=True, positive=True)
    expr = (
        sp.exp(-beta * (x - x1 - (k - 1) * r_u1))
        + sp.exp(-beta * (x2 - x - (k - 1) * r_u2))
        - gamma
    )

    for r_u2_val in range(math.ceil(r_u2_min), math.ceil(r_u2_max) + 1):
        x_mid = (x1 + x2 - (k - 1) * (r_u1 - r_u2_val)) / 2
        inequality_mid = expr.subs({r_u2: r_u2_val, x: x_mid})
        if inequality_mid < 0:
            continue

        inequality_left = expr.subs({r_u2: r_u2_val, x: x2 - k * r_u2_val})
        if inequality_left < 0:
            continue

        inequality_right = expr.subs({r_u2: r_u2_val, x: x1 + k * r_u1})
        if inequality_right < 0:
            continue

        return r_u2_val

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
        # print("r_last = ", r_last)
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


def evolution(population, i, z, z_nad, fitness, f1, f2):
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


def main():
    # init
    manager = multiprocessing.Manager()
    fitness = manager.list([None] * pop_size)
    f1 = manager.list([None] * pop_size)
    f2 = manager.list([None] * pop_size)
    archive_f = manager.list()

    for i in range(pop_size):
        f1[i], f2[i], r[i] = evaluate(population[i])
        # archive_individual.append(population[i])
        # archive_r.append(r[i])
        archive_f.append([f1[i], f2[i]])

    z = [min(f1), min(f2)]
    z_nad = [max(f1), max(f2)]

    for i in range(pop_size):
        fitness[i] = calculate_fitness(f1[i], f2[i], z, z_nad, lamb[i])
        # archive_fitness.append(fitness[i])

    for generation in range(max_generation):
        print("----- generation =", generation)
        process = []
        for i in range(pop_size):
            p = multiprocessing.Process(
                target=evolution,
                args=(
                    population,
                    i,
                    z,
                    z_nad,
                    fitness,
                    f1,
                    f2,
                ),
            )
            process.append(p)
            p.start()

        for p in process:
            p.join()

        z = [min(min(f1), z[0]), min(min(f2), z[1])]
        z_nad = [max(max(f1), z_nad[0]), max(max(f2), z_nad[1])]
        for i in range(pop_size):
            archive_f.append([f1[i], f2[i]])

    # for j in range(len(archive_r)):
    #     print("j = ", j)
    #     print("archive_individual = ", archive_individual[j])
    #     print("archive_fitness = ", archive_fitness[j])
    #     print("archive_r = ", archive_r[j])
    #     print("archive_f = ", archive_f[j])
    for j in range(len(population)):
        print("r = ", r[j])

    aaa = []
    for generation in range(max_generation + 1):
        for j in range(len(population)):
            aaa.append(
                [
                    archive_f[generation * pop_size + j][0] / z_nad[0],
                    archive_f[generation * pop_size + j][1] / z_nad[1],
                ],
            )

    with open(output_file, "w") as file:
        for item in aaa:
            file.write(f"{item[0]} {item[1]}\n")


if __name__ == "__main__":
    time_start = time.time()
    main()
    time_end = time.time()
    print("Time: ", time_end - time_start)
