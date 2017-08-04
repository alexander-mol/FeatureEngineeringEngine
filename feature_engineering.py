import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random
import time


class Individual():
    def __init__(self, genes):
        self.fitness = None
        self.genes = set(genes)

    def score(self, X, y):
        if self.genes is None:
            self.fitness = -np.inf
        else:
            self.fitness = Engine().evaluate(X[list(self.genes)], y, max_time=0.4)

class Engine:

    def __init__(self, clf=None):
        """
        Create a feature engineering engine.
        """
        if clf is None:
            self.clf = LogisticRegression()
        else:
            self.clf = clf

    def fit(self, X, y):
        """
        Engineer features to get the best cross-validation accuracy.
        :param dataframe X: raw features
        :param array-like y: labels vector
        :return: dataframe of engineered features
        """
        # dummify categorical variables
        X = self.dummify_columns(X)
        # square columns with continuous data
        X = self.squares(X)
        # log columns with positive data
        X = self.logs(X)
        # make interaction columns
        # X = self.interactions(X)
        # X = self.simplify(X, y)
        X = self.reduce(X, y)
        return X

    def score(self, X, y):
        return self.evaluate(X, y)

    def dummify_columns(self, X):
        """
        Create dummy columns for categorical variables in X.
        """
        for column in X.columns:
            if isinstance(X[column].iloc[0], str):
                categories = list(X[column].unique())
                for category in categories[:-1]:
                    X[category+'_dummy'] = X[column].apply(lambda x: x == category) * 1
                X.drop(column, 1, inplace=True)
        return X

    def evaluate(self, X, y, max_time=1.0, min_runs=10):
        """
        Spend max_time trying to get a good estimate of the cross-validation accuracy of the model given feature set X.
        """
        if len(X.columns) == 0:
            return -np.inf
        X_s = StandardScaler().fit_transform(X)
        runs = 0
        running_sum = 0
        t0 = time.time()
        while (time.time() < t0 + max_time) or (runs < min_runs):
            runs += 1
            X_train, X_test, y_train, y_test = train_test_split(X_s, y)
            self.clf.fit(X_train, y_train)
            running_sum += self.clf.score(X_test, y_test)
        return running_sum/runs

    def simplify(self, X, y, max_reduction=0.01):
        """
        Remove the least predictive features up to a specified accuracy reduction.
        """
        best_accuracy = self.evaluate(X, y)
        X_new = X.copy()

        while len(X_new.columns) > 1:
            remove_col_index = np.argmin(np.abs(self.clf.coef_))
            X_temp = X_new.drop(X_new.columns[remove_col_index], 1)
            this_accuracy = self.evaluate(X_temp, y)
            if best_accuracy - this_accuracy < max_reduction:
                best_accuracy = max(this_accuracy, best_accuracy)
                print('Removing '+X_new.columns[remove_col_index])
                X_new = X_temp
            else:
                break
        X = X_new
        return X

    def reduce(self, X, y, max_reduction=0.01):
        best_accuracy = self.evaluate(X, y)
        X_new = X.copy()

        change = True
        while len(X_new.columns) > 1 and change:
            change = False
            for col in X_new.columns:
                X_temp = X_new.drop(col, 1)
                this_accuracy = self.evaluate(X_temp, y)
                if best_accuracy - this_accuracy < max_reduction:
                    best_accuracy = max(this_accuracy, best_accuracy)
                    print('Removing ' + col + ', acc: '+str(this_accuracy))
                    X_new = X_temp
                    change = True
                else:
                    break
        X = X_new
        return X

    def squares(self, X):
        """
        Square columns containing continuous data.
        """
        for column in X.columns:
            if len(X[column].unique()) > 2:  # i.e. not dummy variable
                X[column+'^2'] = X[column] ** 2
        return X

    def logs(self, X):
        """
        Take logs of columns with positive values.
        """
        for column in X.columns:
            if len(X[column].unique()) > 2 and column[-2:] != '^2':  # i.e. not dummy variable or already squared
                if np.count_nonzero(X[column] <= 0) == 0:  # check all positive
                    X['log('+column+')'] = np.log(X[column].astype('float64'))
        return X

    def interactions(self, X):
        """
        Make interaction columns using pairwise multiplication.
        """
        X_new = X.copy()
        for i in range(len(X.columns)):
            for j in range(i + 1, len(X.columns)):
                X_new[X.columns[i]+'.'+X.columns[j]] = X[X.columns[i]] * X[X.columns[j]]
        X = X_new
        return X

    # def dummy_or(self, X):
    #     dummy_columns = []
    #     for col in X.columns:
    #         if set(X[col].unique()) == {0, 1}:  # then it's a dummy column
    #             dummy_columns.append(col)
    #     for col1 in dummy_columns:
    #         for col2 in dummy_columns:
    #             X[col1+'+'+col2] = X[col1] | X[col2]


    def genetic(self, X, y):
        X = self.dummify_columns(X)
        X = self.squares(X)
        X = self.logs(X)
        X = self.interactions(X)

        max_pop = 100
        max_gen = 1000
        genes = set(X.columns)
        population = []
        for i in range(max_pop):
            population.append(Individual(random.sample(genes, round(len(genes)/2))))
        half_pop = round(len(population)/2)

        for generation in range(max_gen):
            for individual in population:
                individual.score(X, y)
            population.sort(key=lambda x: x.fitness, reverse=True)
            for i in range(half_pop, len(population)):
                mother = random.choice(population[:half_pop])
                father = random.choice(population[:half_pop])
                population[i] = Individual(random.sample(mother.genes, round(len(mother.genes)/2)) +
                    random.sample(father.genes, round(len(father.genes)/2)) )
                mutations = int(np.log(1.0 / random.random()) / np.log(2))
                try:
                    if random.random() < 0.5:
                        additonal_genes = set(random.sample(genes, mutations))
                        population[i].genes = population[i].genes.union(additonal_genes)
                    else:
                        population[i].genes = population[i].genes - set(random.sample(population[i].genes, mutations))
                except ValueError:
                    pass
            print(population[0].fitness, population[half_pop-1].fitness, population[0].genes)


    def genetic_fast(self, X, y):
        pass



    # def grid_search(self, X, y):
    #     X = self.dummify_columns(X)
    #     X = self.squares(X)
    #     X = self.logs(X)
    #
    #     fields = set(X.columns)
    #     for
    #         self.evaluate(X, y,)
