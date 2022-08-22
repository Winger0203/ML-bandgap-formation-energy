import numpy as np
import random
from sklearn import datasets
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from deap import creator, base, tools, algorithms


def getFitness(individual, X, y):

    if individual.count(0) != len(individual):
        cols = [index for index in range(len(individual)) if individual[index] == 0]
        X_sub = np.delete(X, cols, axis=1)

        clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)

        scores = cross_val_score(clf, X_sub, y, cv=3)
        Average_score = sum(scores)/len(scores)
        return (Average_score,)
    else:
        return(0,)


def geneticAlgorithm(X, y, n_population, n_generation):
    """
    Deap global variables
    Initialize variables to use eaSimple
    """
    # create individual
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # create toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, X.shape[1])
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)
    toolbox.register("evaluate", getFitness, X=X, y=y)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # initialize parameters
    pop = toolbox.population(n=n_population)
    hof = tools.HallOfFame(n_population * n_generation)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # genetic algorithm
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                   ngen=n_generation, stats=stats, halloffame=hof,
                                   verbose=True)

    # return hall of fame
    return hof


def bestIndividual(hof, X, y):

    maxAccurcy = 0.0
    for individual in hof:
        # print('tuple value = ',individual.fitness.values)
        if(individual.fitness.values[0] > maxAccurcy):
            maxAccurcy = individual.fitness.values
            _individual = individual

    _individualHeader = [list(X)[i] for i in range(len(_individual)) if _individual[i] == 1]
    return _individual.fitness.values, _individual, _individualHeader


if __name__ == '__main__':
    # 导入数据
    breast = datasets.load_breast_cancer()
    X = breast.data
    y = breast.target
    print(X.shape)

    # 数据归一化
    mms = preprocessing.MinMaxScaler()
    X = mms.fit_transform(X)

    # get accuracy with all features
    individual = [1 for i in range(X.shape[1])]
    print("Accuracy with all features: \t" + str(getFitness(individual, X, y)) + "\n")

    # apply genetic algorithm
    n_pop = 5
    n_gen = 10
    hof = geneticAlgorithm(X, y, n_pop, n_gen)

    # select the best individual
    accuracy, individual, _ = bestIndividual(hof, X, y)
    print('Best Accuracy: \t' + str(accuracy))
    print('Number of Features in Subset: \t' + str(individual.count(1)))
    print('Individual: \t\t' + str(individual))

    print('\n\ncreating a new classifier with the following selected features:')

    cols = [index for index in range(len(individual)) if individual[index] == 0]
    X_selected = np.delete(X, cols, axis=1)
    selected_cols = [index for index in range(len(individual)) if individual[index] != 0]
    print(selected_cols)
    print(X_selected.shape)

    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)

    scores = cross_val_score(clf, X_selected, y, cv=3)
    Average_score = sum(scores)/len(scores)
    print("Accuracy with Feature Subset: \t" + str(Average_score) + "\n")
