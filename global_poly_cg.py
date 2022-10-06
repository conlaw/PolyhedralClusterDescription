from global_poly_pricing import GlobalPolyClusterExplanationPP01
from global_poly_master import GlobalPolyClusterExplanationSolver
import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
import time


class GlobalPolyClusterExplainCG(object):
    '''
    Object to create a cluster explanation using column gen
    '''
    
    def __init__(self, X, clusters, W, b, bad_hps = 0, alpha = 0, alpha_type = 'global', M = 1, beta = 1, 
                 objective='num_hp', colGen_allowed = True, new_hp_prefix = 'CG_'):
        
        #Define class variables
        self.X = X
        self.clusters = clusters
        self.W = W
        self.b = b
        self.alpha = alpha
        self.alpha_type = alpha_type
        self.objective = objective
        self.bad_hps = bad_hps
        self.colGen_allowed = colGen_allowed
        self.numIter = 0 
        self.colCounter = 0
        self.new_hp_prefix = new_hp_prefix
        
        # Map parameters to instantiated objects
        self.master = GlobalPolyClusterExplanationSolver(self.X, self.clusters, self.W, self.b, self.alpha, self.alpha_type,  self.bad_hps, objective = objective)
        
        if self.colGen_allowed:
            self.pricing = {}
            for cl in np.unique(self.clusters):
                self.pricing[cl] = GlobalPolyClusterExplanationPP01(self.X, self.clusters,  cl, M = M, beta = beta, objective = objective)
        
    def solve(self, verbose = False, colGen = True, timeLimit = 300):
        if not self.colGen_allowed:
            print('col gen disabled for this object.')
            colGen = False
        
        if timeLimit is not None:
            start_time = time.perf_counter()
        
        lp_time = 0
        pricing_time = 0
                
        while colGen:
            self.numIter += 1
            # Solve relaxed version of restricted problem
            if verbose:
                print('Solving Restricted LP')
            
            lp_solve = time.perf_counter()
            results = self.master.solve(verbose = verbose, relax = True)
            print(results['m'].objVal)
            lp_time += time.perf_counter() - lp_solve

            # Generate new candidate rules
            if verbose:
                print('Solving Pricing')
            
            new_hp_flag = False
            #print(results)
            for cl in results['duals']['mu_prime'].keys():
                pricing_iter = time.perf_counter()
                pricing_res = self.pricing[cl].solve(results['duals']['mu_self'][cl], 
                                                 results['duals']['mu_prime'][cl],
                                                 results['duals']['gamma'],verbose=False)
                pricing_time += time.perf_counter() - pricing_iter
                #print('Objective Value of Pricing %d:'%cl, pricing_res['objVal'])
                
                if pricing_res['newHP']:
                    new_hp_flag = True
                    for key in pricing_res['w'].keys():
                        self.colCounter += 1
                        #print('Adding '+'CG_new'+str(self.colCounter))
                        #print(pricing_res['w'], pricing_res['b'])
                        self.W[cl][self.new_hp_prefix+str(self.colCounter)] = pricing_res['w'][key]
                        self.b[cl][self.new_hp_prefix+str(self.colCounter)] = pricing_res['b'][key]

           #print('Objective Value of Pricing:', pricing_res['objVal'])
            # If no new rules generated exit out and solve master to optimality
            if new_hp_flag:
                #print(pricing_res['w'], pricing_res['b'])
                if verbose:
                    print('Adding new columns')
                
                self.master = GlobalPolyClusterExplanationSolver(self.X, self.clusters, self.W, self.b, self.alpha, self.alpha_type, self.bad_hps, objective = self.objective)

            else:
                break
                
            if timeLimit is not None: 
                if time.perf_counter() - start_time > timeLimit:
                    print('Time limit for column generation exceeded. Solving MIP.')
                    break
                            
        
        # Solve master problem to optimality
        if verbose:
            print('Solving final master problem to integer optimality')
        
        ip_solve_time = time.perf_counter()
        results = self.master.solve(verbose = verbose, relax = False)
        
        if results['status'] == 'FAILED':
            return results
        else:
            print(results['m'].objVal)
            results['objective'] = results['m'].objVal
            ip_solve_time = time.perf_counter() - ip_solve_time
            results['ip_solve_time'] = ip_solve_time
            results['num_iter'] = self.numIter
            results['lp_time'] = lp_time
            results['pricing_time'] = pricing_time
                
        #Return final rules
        return results
        
