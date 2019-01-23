import pandas as pd
import numpy as np
# from __future__ import print_function

import sys

import cplex
from cplex.exceptions import CplexSolverError
from cplex import SparsePair
from cplex.six.moves import zip

def General_CG(w):
    
    M = cplex.Cplex()

    var=list(range(len(w)))

    M.variables.add(obj=[1]*len(var))

    M.linear_constraints.add(lin_expr=[SparsePair()] * len(w),
                               senses=["G"] * len(w),
                               rhs=[1] * len(w))
    for i in range(len(w)):
        M.linear_constraints.set_coefficients(i,i, 1)

    M.objective.set_sense(M.objective.sense.minimize)    


    S = cplex.Cplex()

    S.variables.add(types=[S.variables.type.integer] * len(w),obj=[1]*len(w))

    totalsize = SparsePair(ind=list(range(len(w))), val= w )
    S.linear_constraints.add(lin_expr=[totalsize],
                               senses=["L"],
                               rhs=[150])

    S.objective.set_sense(S.objective.sense.maximize)


    ite=0


    while True:
        ite+=1
        M.write('m.lp')

        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)
        M.solve()

        price = [pie for pie in M.solution.get_dual_values(list(range(len(w))))]
        S.objective.set_linear(list(zip(list(range(len(w))), price)))
    #     S.write('s.lp')
        S.set_log_stream(None)
        S.set_error_stream(None)
        S.set_warning_stream(None)
        S.set_results_stream(None)
        S.solve()

        if S.solution.get_objective_value() < 1+1.0e-6:
            break
            
        newsub = S.solution.get_values()

        idx = M.variables.get_num()
        M.variables.add(obj=[1.0])
        M.linear_constraints.set_coefficients(list(zip(list(range(len(w))),
                                                         [idx] * len(var),
                                                         newsub)))
        var.append(idx)

    M.variables.set_types(
        list(zip(var, [M.variables.type.integer] * len(var))))
    M.solve()
    
    return ite,M,S


def Chebyshev_CG(w):

    ######### Master Problem ###########

    M = cplex.Cplex()

    # Parameters
    var = list(range(len(w)))
    alpha = 1.0
    init_pi = sum(w)/150
    epsilon = 0.1


    # decision varialbes types=["C"]*len(var) 
    M.variables.add(obj=[1]*len(var), names=['x_'+str(i) for i in var])
    M.variables.add(obj=[-init_pi], names='z')
    M.variables.add(names = ['y_'+str(i) for i in list(range(len(w)))])


    # pattern constraints
    vals = np.zeros((len(w), len(var)))

    np.fill_diagonal(vals, 1)

    M.linear_constraints.add(
        lin_expr= [
            cplex.SparsePair(
                ind =['x_'+str(j) for j in var]+['y_'+str(i)]+['z'] , 
                val = list(vals[i]) + [-1.0] + [-1.0] 
                )
        for i in range(len(w))
        ],
        senses=["G" for i in w],
        rhs=[0 for i in w] )

    # chebyshev constraint
    M.linear_constraints.add(
        lin_expr= [
            cplex.SparsePair(
                ind =['x_'+str(j) for j in var]+['y_'+str(i) for i in range(len(w))]+['z'] , 
                val = [1.0 for k in var]  + [1.0 for l in w] + [alpha*120**(1/2)]
                )
        ],
        senses=["G"],
        rhs=[1.0] )


    #list(np.linalg.norm(vals,1))


    # set objective
    M.objective.set_sense(M.objective.sense.minimize)    


    ######### Separation Problem ###########
    S = cplex.Cplex()

    S.variables.add(types=[S.variables.type.integer] * len(w),obj=[1]*len(w))

    totalsize = SparsePair(ind=list(range(len(w))), val= w )
    S.linear_constraints.add(lin_expr=[totalsize],
                               senses=["L"],
                               rhs=[150])

    S.objective.set_sense(S.objective.sense.maximize)


    ite=0
    while True:
        ite+=1
        M.write('cheby_m.lp')

        M.set_log_stream(None)
        M.set_error_stream(None)
        M.set_warning_stream(None)
        M.set_results_stream(None)
        M.solve()

        price = [pie for pie in M.solution.get_dual_values(list(range(len(w))))]
        S.objective.set_linear(list(zip(list(range(len(w))), price)))
        S.write('cheby_s.lp')
        S.set_log_stream(None)
        S.set_error_stream(None)
        S.set_warning_stream(None)
        S.set_results_stream(None)
        S.solve()


        if M.solution.get_objective_value() < epsilon * M.solution.get_values('z') :
            break

        if S.solution.get_objective_value() < 1+1.0e-6:
            newsub = S.solution.get_values()
            idx = M.variables.get_num()
            M.variables.add(obj=[1.0])
            M.linear_constraints.set_coefficients(list(zip(list(range(len(w))),
                                                             [idx] * len(var),
                                                             newsub)))
            var.append(idx)


        else :
            new_pi = M.solution.get_dual_values()
            M.objective.set_linear('z',-sum(new_pi))

    M.variables.set_types(
        list(zip(var, [M.variables.type.continuous] * len(var))))
    M.solve()

    return ite, M,S




# data preprocessing
test = pd.read_csv("test.txt",sep="\n",header=None) 

ind = list(test[test[0].str.contains("u")].index)

probs = [[int(test[0][i]) for i in range(ind[j]+2,ind[j+1])] for j in range(len(ind)-1)]
probs.append([int(test[0][i]) for i in range(ind[len(ind)-1]+2,len(test[0]))])



# execution
num = 0

w = probs[num]

ite , M, S = General_CG(w)

C_ite, C_M, C_S = Chebyshev_CG(w)

