import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

#Methods to give back dual values in relax mode

class GlobalPolyClusterExplanationSolver(object):
    
    def __init__(self, X, clusters, W, b, alpha, alpha_type = 'global', bad_hps = 0, beta = None, M = None, 
                 objective='num_hp', error_type='binary'):
        
        #Save constants
        self.X = X
        self.clusters = clusters
        self.cluster_unique = np.unique(self.clusters)
        self.k = len(self.cluster_unique)
        self.W = W
        self.b = b
        self.alpha = alpha
        self.alpha_type = alpha_type
        self.beta = beta if beta is not None else X.shape[1]
        self.M = M if M is not None else len(W.keys())
        
        self.obj_type = objective      
        self.error_type = error_type
        self.bad_hps = bad_hps
        
        #Initialize model
        self.init_model()
        
    
    def init_model(self):
        #Create gurobi model object
        self.m = gp.Model("global_cluster_explanation")
        
        #Initialize model's variables, consraints and objectives
        self.init_dv()
        self.add_constr()
        self.set_obj()
        
        #Update the model to save all the changes
        self.m.update()
        
    def solve(self, verbose = False, relax = False):
        # Set time limit
        #self.m.Params.TimeLimit = 300
        self.m.update()

        #Set how chatty we want our results to be
        self.m.Params.OutputFlag = verbose
        
        #Solve the optimization model
        if relax:
            m = self.m.relax()
        else:
            m = self.m
        
        m.optimize()
        
        if m.Status == 2:
        
            #Return various metrics about the run
            return {
                'status': 'SOLVED',
                'num_constr': m.NumConstrs,
                'num_vars': m.NumVars,
                'b&b_nodes': m.NodeCount if not relax else -1,
                'time': m.Runtime,
                'z': np.array([dv.x for dv in m.getVars()[(self.X.shape[0]+self.X.shape[1]):]]),
                'y': np.array([dv.x for dv in m.getVars()[self.X.shape[0]:(self.X.shape[0]+self.X.shape[1])]]),
                'eps': np.array([dv.x for dv in m.getVars()[:self.X.shape[0]]]),
                'duals': None if not relax else self.getDuals(m),
                'm' : m,
                'obj': m.objVal,
            }
        
        else:
            return {
                'status': 'FAILED'
            }

    
    def getDuals(self, m):
        mu_prime = {}
        mu_self = {}
        
        if self.obj_type != 'feasibility':
            start_const = 1 if self.alpha_type == 'global' else len(self.cluster_unique)
        else:
            start_const = 0 if self.alpha_type == 'global' else len(self.cluster_unique) 
        
        for k in self.cluster_unique:
            X_sub = self.X[self.clusters != k]
            end_const = start_const + X_sub.shape[0]
            mu_prime[self.cluster_unique[k]] = np.array([const.Pi for const in m.getConstrs()[start_const:end_const]])
            start_const = end_const
            
        for k in self.cluster_unique:
            X_cl = self.X[self.clusters == k]
            end_const = start_const + X_cl.shape[0]
            mu_self[self.cluster_unique[k]] = np.array([const.Pi for const in m.getConstrs()[start_const:end_const]])
            start_const = end_const

        gamma = np.array([const.Pi for const in m.getConstrs()[start_const:(start_const+X_sub.shape[1])]])
        
        return {'mu_prime': mu_prime,'mu_self': mu_self, 'gamma': gamma}
        
    def init_dv(self):
        #Create eps vars to track eps
        if self.error_type == 'binary':
            self.eps = self.m.addMVar(self.X.shape[0], vtype = gp.GRB.BINARY, name='EPS')
        elif self.error_type == 'hamming':
            self.eps = self.m.addMVar(self.X.shape[0], vtype = gp.GRB.INTEGER, name='EPS')
        
        #create y vars
        self.y = self.m.addMVar(self.X.shape[1], vtype = gp.GRB.BINARY, name='y')

        #Creat z vars
        self.z = {}
        self.z_complex = {}
        for cl in self.cluster_unique:
            self.z[cl] = self.m.addMVar(len(self.W[cl].keys()), lb = 0, vtype = gp.GRB.INTEGER, name='Z_%d'%(cl))
            self.z_complex[cl] = np.array((np.array(list(self.W[cl].values())) != 0).sum(axis=1) + 1)
        
    def generateZMatrix(self, k):
        X_sub = self.X[self.clusters != k]
        
        Z = np.zeros((X_sub.shape[0],len(self.W[k].keys())))
        if len(self.W[k].keys())== 0:
            print('Cluster %d has no init halfspaces'%k)
        
        for id, w in enumerate(self.W[k].keys()):
            Z[:,id] = ((X_sub @ np.array(self.W[k][w]).T).values.reshape(-1) + self.b[k][w] > 0).astype(np.int)
        
        return Z
    
    def generateZ2Matrix(self, k):
        X_sub = self.X[self.clusters == k]
        
        Z = np.zeros((X_sub.shape[0],len(self.W[k].keys())))
        for id, w in enumerate(self.W[k].keys()):
            Z[:,id] = ((X_sub @ np.array(self.W[k][w]).T).values.reshape(-1) + self.b[k][w] > 0).astype(np.int)
        
        return Z

    def generateYMatrix(self):
        Y = {}
        for cl in self.cluster_unique:
            Y_cl = np.zeros((self.X.shape[1],len(self.W[cl].keys())))                                        
            for id, w in enumerate(self.W[cl].keys()):
                Y_cl[:,id] = (self.W[cl][w] != 0).astype(np.int)
            Y[cl] = Y_cl
        return Y

    def add_columns(self, W_new, b_new): 
        '''
        Function to add new rules to the restricted model.
        -Input takes LIST of rule objects
        '''
        self.m.update()
        
        new_cols = len(W_new.keys())
        
        for key in W_new.keys():
            
            #Specify new column
            newCol = gp.Column()
            
            start_const = 1
            for k in range(len(self.cluster_unique)):
                X_sub = self.X[self.clusters != k]
                end_const = start_const + X_sub.shape[0]

                col_vals = ((X_sub @ np.array(W_new[key]).T).values.reshape(-1) + b_new[key]   >= 0).astype(np.int).reshape(-1)
                
                newCol.addTerms(col_vals , self.m.getConstrs()[start_const:(end_const)])
                start_const = end_const
                
            for k in range(len(self.cluster_unique)):
                X_cl = self.X[self.clusters == k]
                end_const = start_const + X_cl.shape[0]

                col_vals = ((X_cl @ np.array(W_new[key]).T).values.reshape(-1) + b_new[key]   >= 0).astype(np.int).reshape(-1)
                
                newCol.addTerms(col_vals , self.m.getConstrs()[start_const:(end_const)])
                start_const = end_const

            newCol.addTerms((W_new[key][0] != 0).astype(np.int) , 
                            self.m.getConstrs()[start_const:(start_const+self.X.shape[1])])
                        
            #Add decision variable
            self.m.addVar(obj=1 if self.obj_type == 'num_hp' else 0, 
                              vtype=GRB.INTEGER,
                              lb = 0,
                              name="z[%d]"%(len(self.W.keys())+1), 
                              column=newCol)
            
            self.W[key] = W_new[key]
            self.b[key] = b_new[key]
        

    def add_constr(self):
        #Add FPR constraint
        self.alpha_constrs = 0
        
        if self.obj_type != 'feasibility':
            if self.alpha_type == 'global':
                self.alpha_constr = self.m.addConstr(self.eps.sum() <= self.alpha, name='alpha')
                self.alpha_constrs += 1
            elif self.alpha_type == 'local':
                self.alpha_constr = {}
                for cl in self.cluster_unique:
                    self.alpha_constr[cl] = self.m.addConstr(self.eps[self.clusters == cl].sum() <= self.alpha[cl], name='alpha_%d'%cl)
                    self.alpha_constrs += 1
        else:
            if self.alpha_type == 'local':
                self.alpha_constr = {}
                for cl in self.cluster_unique:
                    self.alpha_constr[cl] = self.m.addConstr(self.eps[self.clusters == cl].sum() <= self.alpha[cl], name='alpha_%d'%cl)
                    self.alpha_constrs += 1
        #print(self.alpha_constrs, ' alpha constraints added')
            
        #Add constraint to ensure we meet item demand
        for id,cl in enumerate(self.cluster_unique):
            Z = self.generateZMatrix(cl)
            self.m.addConstr(Z @ self.z[cl] + self.eps[self.clusters != cl] >= 1,name='set_cover_k%d'%cl)
            
        #Add constraint to ensure we meet item demand
        for id, cl in enumerate(self.cluster_unique):
            Z = self.generateZ2Matrix(cl)
            if self.error_type == 'binary':
                self.m.addConstr(- Z @ self.z[cl] + self.M*self.eps[self.clusters == cl] >= 0,name='set_cover_k_self%d'%cl)
            elif self.error_type == 'hamming':
                self.m.addConstr(- Z @ self.z[cl] + self.eps[self.clusters == cl] >= 0,name='set_cover_k_self%d'%cl)


        Y = self.generateYMatrix()
        self.m.addConstr(sum([Y[k] @ self.z[k] for k in self.cluster_unique]) - self.M * self.y <= 0, name='feature_y')
        
        if self.obj_type != 'sparsity':
            self.m.addConstr(self.y.sum() <= self.beta, name='sparsity')
        
        
    def set_obj(self):
        #Minimize # of hps used
        if self.obj_type == 'num_hp':
            if self.bad_hps > 0:
                self.m.setObjective(1000*self.z[:self.bad_hps].sum() + self.z[self.bad_hps:].sum(), gp.GRB.MINIMIZE)
            else:
                self.m.setObjective(sum([self.z[k].sum() for k in self.cluster_unique]), gp.GRB.MINIMIZE)
        
        elif self.obj_type == 'complex':
            obj = 0
            for k in self.cluster_unique:
                obj += self.z_complex[k] @ self.z[k]
            self.m.setObjective(obj, gp.GRB.MINIMIZE)

        elif self.obj_type == 'sparsity':
            self.m.setObjective(self.y.sum(), gp.GRB.MINIMIZE)
            
        elif self.obj_type == 'feasibility':
            if self.bad_hps > 0:
                self.m.setObjective(self.X.shape[0]*self.z[:self.bad_hps].sum() + self.eps.sum(), gp.GRB.MINIMIZE)
            else:
                self.m.setObjective(self.eps.sum(), gp.GRB.MINIMIZE)
