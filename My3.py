import pandas as pd
import numpy as np
import csv
import xlrd
import random
import math
import copy
import sys
import tsp
import timeit



class solution:
    def __init__(self, dist_mat, route=None):
        if route is None:
            route = []
        self.route = route
        self.cost = int(self.total_cost(dist_mat))

    def total_cost(self, dist_mat):
        cost = 0
        for i in range(len(self.route) - 1):
            cost += self.length(self.route[i], self.route[i + 1], dist_mat)

        self.cost = cost

        return cost

    def length(self, n1, n2, mat):
        return mat[n1][n2]

    def __delete__(self):
        del self

def two_opt_step(route, rev_start, rev_end):
    for i in range(int((rev_end-rev_start+1)/2)):
        temp = route[rev_start+i]
        route[rev_start + i] = route[rev_end-i]
        route[rev_end - i] = temp

    return route



def NearestNeighbour(df):
    """
    This function creates an initial solution from a distance matrix
    df: distance_matrix
    """
    bigM = 9999999
    route = []
    route.append(0) # initilaze route with 0

    current_position = route[0]
    df.iloc[[route[0]]] = bigM # close down the starting point

    for i in range(0,len(df.columns)-1):
        route.append(int(df[df[current_position] == df[current_position].min()].index.values.flat[0]))
        df.iloc[[route[-1]]] = bigM # so that one will not return to visited position
        current_position = route[-1] # update position

    return route





def SimulatedAnnealing(initial):
    global df
    temperature = initial.cost * 10
    cool_rate = 0.995
    stagnation_counter = 0

    print("Initial: ", initial.cost)
    # Start of the Algorithm so the best and current solution is the initial solution
    current = copy.deepcopy(initial)
    best = copy.deepcopy(initial)

    # create a bag that with possible two opt routes
    visited_list = createBag(len(current.route))

    while temperature > 0.01 and stagnation_counter < 10000:
        # TODO: write code for the case that there are no possible options left for two-opt: LOCAL MINIMA
        # choose one of the remaining possibilities
        two_opt, visited_list = randBag(visited_list)

        # CREATING a new route, two_opt_step modifies the current solution
        candidate = solution(df, two_opt_step(copy.deepcopy(current.route), two_opt[0], two_opt[1]))

        if candidate.cost < best.cost:
            best = copy.deepcopy(candidate)

        # Are we in a shorter route?
        epsilon = candidate.cost - current.cost




        if epsilon >= 0:
            # Candidate is not better so we count that
            stagnation_counter += 1



            # TODO: variable prob is for debugging remove later
            prob = math.exp(-epsilon/temperature)
            # TODO: variable a is for debugging remove later
            a = random.uniform(0, 1)
            if prob > a:
                print("!! Worse Solution !!              ", (stagnation_counter, current.cost))
                current = candidate
                candidate.__delete__()
                visited_list = createBag(len(current.route)) # refresh the list
                temperature *= cool_rate  # At each step my tolerance for bad solution should decrease
            else:
                # TODO Remove this afterwards
                candidate.__delete__()
                print("!! Worse Solution !! NO CHANGE !!", (stagnation_counter, current.cost))
        elif epsilon < 0:
            print("-- Better --")
            # Better candidate was found! Refresh the counter
            stagnation_counter = 0
            visited_list = createBag(len(current.route))  # refresh the list

            current = candidate
            # deleting the candidate just in case it is used somewhere
            candidate.__delete__()


    print("Best: ", best.cost)
    print("Best: ", best.route)

    best.route.append(best.route[0])
    best.total_cost(df)

    print("Best: ", best.cost)
    print("Best: ", best.route)



"""These are functions for chosing 2-opt cities"""

def createBag(n_max):
    bag = []
    for i in range(n_max):
        for j in range(n_max):
            if i != j and j > i:
                bag.append((i,j))

    return bag

def randBag(list):
    #random.seed(333)
    element_index = random.randint(0, len(list)-1)
    element = list[element_index]
    list.remove(element)
    return element, list


def randPoints(n_max, list):
    #random.seed(333)
    x = random.randint(0, n_max - 2)
    y = random.randint(x+1, n_max - 1)

    while [x, y] in list:
        #random.seed(333)
        x = random.randint(0, n_max - 2)
        y = random.randint(x + 1, n_max - 1)

    list.append([x,y])
    return x, y, list

"""End of functions for chosing 2-opt cities"""

def main():
    """ This is the main story """
    global df
    start = timeit.timeit()
    in_sol = solution(df, NearestNeighbour(copy.deepcopy(df)))
    SimulatedAnnealing(in_sol)
    stop = timeit.default_timer()

    print('Time: ', stop - start)
# readi ng the distance matrix
df = pd.read_excel(r'/Users/gokhanmakaraci/Downloads/asimetrik_data_40.xlsx',sheet_name='20') #(use "r" before the path string to address special character, such as '\'). Don't forget to put the file name at the end of the path + '.xlsx'
# df = read_TSPLIB("/Users/gokhanmakaraci/Downloads/ftv44.atsp", precision = 0)
if __name__ == "__main__":
    main()
