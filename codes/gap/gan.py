import pandas as pd
import numpy as np
# from __future__ import print_function

import sys

import cplex
from cplex.exceptions import CplexSolverError
from cplex import SparsePair
from cplex.six.moves import zip

def data(problem):
    
    f = open("gap_d/%s.txt"%(problem), 'r')
    data = f.readlines()
    f.close()

    records = []
    for line in data:
        record = [int(field) for field in line.strip().lstrip('[').rstrip(']').split()]
        records.append(record)

    size = records[0]
    agent = size[0]
    job = size[1]

    c = []
    a = []
    b = []
    for i in range(len(records)-1) :
        if len(c) < job*agent:
            c.extend(records[i+1])
        elif len(c) >= job*agent and len(a)< job*agent :
            a.extend(records[i+1])
        else :
            b.extend(records[i+1])

    c = np.array(c,dtype=int).reshape((agent,job))
    a = np.array(a,dtype=int).reshape((agent,job))
    b = np.array(b)
    
    return a,b,c,agent,job

def General_CG(a,b,c,agent,job):
    K = range(1)
    var = list(range(agent))

    M = cplex.Cplex()

    x_i_k = lambda i,k: 'x_%d_%d' % (i,k)
    x = [x_i_k(i,k) for i in range(1) for k in K]

    dummy = float(sum(np.sum(c,axis=1)))

    M.variables.add(
        lb = [0] * len(x),
        ub = [1] * len(x),
        names = x,
        obj =  [dummy],
        types = ['C'] * len(x)
    )



    M.linear_constraints.add(
        lin_expr= [
            cplex.SparsePair(
                ind =[x_i_k(i,k) for i in range(1) for k in K], 
                val = [1.0]
                )
        for j in range(job)
        ],
        senses=["G" for j in range(job)],
        rhs=[1.0 for j in range(job)] ,
        names=['assignment_%d' % (j) for j in range(job)])


    M.linear_constraints.add(
        lin_expr= [
            cplex.SparsePair(
                ind =[x_i_k(0,k) for k in K], 
                val = [0] * len(K)
                )
        for i in range(agent)
        ],
        senses=["L" for i in range(agent)],
        rhs=[1.0 for i in range(agent)] )


    M.objective.set_sense(M.objective.sense.minimize)    



    S = cplex.Cplex()

    pi = np.min(c,axis=0)


    d_j = lambda j: 'd_%d' % (j)
    d = [d_j(j) for j in range(job)]


    S.variables.add(
        obj = [int(val) for val in list(pi - np.sum(c,axis=0))],
        types = ['B'] * len(d),
        names = d
    )


    S.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(
                ind = [d_j(j) for j in range(job)], 
                val = [int(v) for v in list(np.sum(a,axis=0))]
                )
        for i in range(agent)
        ],
        senses = ["L" for i in range(agent)],
        rhs = [int(b[i]) for i in range(agent)] )


    S.objective.set_sense(S.objective.sense.maximize)


    ite=0

    criterion = True

    while criterion:

        criterion = False

        for ag in range(agent):   

            ite+=1
            M.write('m.lp')
            M.set_log_stream(None)
            M.set_error_stream(None)
            M.set_warning_stream(None)
            M.set_results_stream(None)
            M.set_problem_type(M.problem_type.LP)
            M.solve()

            pi = list(M.solution.get_dual_values())[0:job]

            S.objective.set_linear(list(zip(list(range(len(d))),list(np.array(pi) -np.array(c[ag])))))
            S.write('s%d.lp'%(ag))
            S.set_log_stream(None)
            S.set_error_stream(None)
            S.set_warning_stream(None)
            S.set_results_stream(None)
            S.solve()

            if S.solution.get_objective_value()-0.000001 > -list(M.solution.get_dual_values())[job+ag]:
                criterion = True
                newsub = S.solution.get_values()
                idx = M.variables.get_num()
                M.variables.add(obj=[np.array(S.solution.get_values()).T @ c[ag]])
                M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                                 [idx] * job,
                                                                 newsub)))

                M.linear_constraints.set_coefficients(job+ag, idx, 1.0)
                var.append(idx)



    M.set_problem_type(M.problem_type.LP)
    M.solve()
    
    return ite,M,S

def Stabilization(a,b,c,agent,job):
    
    K = range(1)
    var = list(range(agent))
    eps = 0.001

    M = cplex.Cplex()

    x_i_k = lambda i,k: 'x_%d_%d' % (i,k)
    x = [x_i_k(i,k) for i in range(1) for k in K]

    dummy = float(sum(np.sum(c,axis=1)))

    M.variables.add(
        lb = [0] * len(x),
        ub = [1] * len(x),
        names = x,
        obj =  [dummy],
        types = ['C'] * len(x)
    )


    gp_j = lambda j: 'gp_%d' % (j)

    gp = [gp_j(j) for j in range(job)]

    M.variables.add(
        lb = [0] * len(gp),
        ub = [eps] * len(gp),
        names = gp,
        obj =  [0] * len(gp),
        types = ['C'] * len(gp)
    )


    gm_j = lambda j: 'gm_%d' % (j)

    gm = [gm_j(j) for j in range(job)]

    M.variables.add(
        lb = [0] * len(gm),
        ub = [eps] * len(gm),
        names = gm,
        obj =  [0] * len(gm),
        types = ['C'] * len(gm)
    )

    yp_i = lambda i: 'yp_%d' % (i)

    yp = [yp_i(i) for i in range(agent)]

    M.variables.add(
        lb = [0] * len(yp),
        ub = [eps] * len(yp),
        names = yp,
        obj =  [0] * len(yp),
        types = ['C'] * len(yp)
    )

    ym_i = lambda i: 'ym_%d' % (i)

    ym = [ym_i(i) for i in range(agent)]

    M.variables.add(
        lb = [0] * len(ym),
        ub = [eps] * len(ym),
        names = ym,
        obj =  [0] * len(ym),
        types = ['C'] * len(ym)
    )



    M.linear_constraints.add(
        lin_expr= [
            cplex.SparsePair(
                ind = x + gp + gm, 
                val = [1.0] * len(K) + [1.0]*len(gp) + [-1.0]*len(gm)
                )
        for j in range(job)
        ],
        senses=["G" for j in range(job)],
        rhs=[1.0 for j in range(job)] ,
        names=['assignment_%d' % (j) for j in range(job)])


    M.linear_constraints.add(
        lin_expr= [
            cplex.SparsePair(
                ind = x + yp + ym , 
                val = [0] * len(K) + [1.0]*len(yp) + [-1.0]*len(ym)
                )
        for i in range(agent)
        ],
        senses=["L" for i in range(agent)],
        rhs=[1.0 for i in range(agent)] )


    M.objective.set_sense(M.objective.sense.minimize)    



    S = cplex.Cplex()

    pi = np.min(c,axis=0)


    d_j = lambda j: 'd_%d' % (j)
    d = [d_j(j) for j in range(job)]


    S.variables.add(
        obj = [int(val) for val in list(pi - np.sum(c,axis=0))],
        types = ['B'] * len(d),
        names = d
    )


    S.linear_constraints.add(
        lin_expr = [
            cplex.SparsePair(
                ind = [d_j(j) for j in range(job)], 
                val = [int(v) for v in list(np.sum(a,axis=0))]
                )
        for i in range(agent)
        ],
        senses = ["L" for i in range(agent)],
        rhs = [int(b[i]) for i in range(agent)] )


    S.objective.set_sense(S.objective.sense.maximize)


    ite=0

    criterion = True

    while criterion:

        criterion = False

        for ag in range(agent):   

            ite+=1
            M.write('m.lp')
            M.set_log_stream(None)
            M.set_error_stream(None)
            M.set_warning_stream(None)
            M.set_results_stream(None)
            M.set_problem_type(M.problem_type.LP)
            M.solve()

            pi = list(M.solution.get_dual_values())[0:job]
            phi =  list(M.solution.get_dual_values())[job:]

            S.objective.set_linear(list(zip(list(range(len(d))),list(np.array(pi) -np.array(c[ag])))))
            S.write('s%d.lp'%(ag))
            S.set_log_stream(None)
            S.set_error_stream(None)
            S.set_warning_stream(None)
            S.set_results_stream(None)
            S.solve()

            if S.solution.get_objective_value()-0.000001 > -list(M.solution.get_dual_values())[job+ag] :
                criterion = True

                M.objective.set_linear(zip(gp+gm+yp+ym , pi+list(-np.array(pi))+phi+list(-np.array(phi))))

                newsub = S.solution.get_values()
                idx = M.variables.get_num()
                M.variables.add(obj=[np.array(S.solution.get_values()).T @ c[ag]])
                M.linear_constraints.set_coefficients(list(zip(list(range(job)),
                                                                 [idx] * job,
                                                                 newsub)))

                M.linear_constraints.set_coefficients(job+ag, idx, 1.0)
                var.append(idx)



    M.set_problem_type(M.problem_type.LP)
    M.solve()

    return ite,M,S



######## execution ########    
problem = 'd05100'

a,b,c,agent,job = data(problem)

ite,M,S = General_CG(a,b,c,agent,job)
ite_s,M_s,S_s = Stabilization(a,b,c,agent,job)