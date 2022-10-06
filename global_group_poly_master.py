import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

#Methods to give back dual values in relax mode

class GlobalGroupPolyClusterExplanationSolver(object):
    
    def __init__(self, X, clusters, weights, W, b, alpha, alpha_type = 'global', bad_hps = 0, beta = None, M = None, objective='num_hp'):
        
        #Save constants
        self.X = X
        self.clusters = clusters
        self.cluster_unique = np.unique(self.clusters)
        self.weights = weights
        self.k = len(self.cluster_unique)
        self.W = W
        self.b = b
        self.alpha = alpha
        self.alpha_type = alpha_type
        self.beta = beta if beta is not None else X.shape[1]
        self.M = M if M is not None else len(W.keys())
        
        self.obj_type = objective        
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
        
        z = np.array([dv.x for dv in m.getVars()[(self.X.shape[0] + int(self.X.shape[1]/2)):]])
        W_final, B_final = self.getSelectedW(z)
        #Return various metrics about the run
        return {
            'num_constr': m.NumConstrs,
            'num_vars': m.NumVars,
            'b&b_nodes': m.NodeCount if not relax else -1,
            'time': m.Runtime,
            'z': z,
            'y': np.array([dv.x for dv in m.getVars()[self.X.shape[0]:int(self.X.shape[1]/2)]]),
            'W_final': W_final,
            'B_final': B_final,
            'eps': np.array([dv.x for dv in m.getVars()[:self.X.shape[0]]]),
            'duals': None if not relax else self.getDuals(m),
            'm' : m,
            'objective': m.objVal
        }
    
    def getSelectedW(self, z):
        W_final = {}
        B_final = {}
        starting_idx = 0
        
        for cl in self.cluster_unique:
            W_final[cl] = {}
            B_final[cl] = {}
            num_hps = len(self.W[cl].keys())

            for key in np.array(list(self.W[cl].keys()))[np.array(z[starting_idx:(starting_idx+num_hps)]).astype(bool)]:
                W_final[cl][key] = self.W[cl][key]
                B_final[cl][key] = self.b[cl][key]
            
            starting_idx = starting_idx + num_hps
            
        return W_final, B_final
            
    
    def getDuals(self, m):
        mu_prime = {}
        mu_self = {}
        
        if self.obj_type != 'feasibility':
            start_const = 1 if self.alpha_type == 'global' else len(self.cluster_unique)
        else:
            start_const = 0
        
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

        gamma = np.array([const.Pi for const in m.getConstrs()[start_const:(start_const+int(X_sub.shape[1]/2))]])
        
        return {'mu_prime': mu_prime,'mu_self': mu_self, 'gamma': gamma}
        
    def init_dv(self):
        #Create eps vars to track eps
        self.eps = self.m.addMVar(self.X.shape[0], vtype = gp.GRB.BINARY, name='EPS')
        
        #create y vars
        self.y = self.m.addMVar(int(self.X.shape[1]/2), vtype = gp.GRB.BINARY, name='y')

        #Creat z vars
        self.z = {}
        for cl in self.cluster_unique:
            self.z[cl] = self.m.addMVar(len(self.W[cl].keys()), lb = 0, vtype = gp.GRB.INTEGER, name='Z_%d'%(cl))
        
        
    def generateZMatrix(self, k):
        X_sub = self.X[self.clusters != k]
        
        Z = np.zeros((X_sub.shape[0],len(self.W[k].keys())))
        if len(self.W[k].keys())== 0:
            print('Cluster %d has no init halfspaces'%k)
        
        for id, w in enumerate(self.W[k].keys()):
            w_concat = np.concatenate([-np.clip(-1*np.array(self.W[k][w]).T.reshape(-1),0,None),
                                       np.clip(np.array(self.W[k][w]).T.reshape(-1),0,None)])

            Z[:,id] = (X_sub @ w_concat + self.b[k][w] > 0).astype(np.int)
        
        return Z
    
    def generateZ2Matrix(self, k):
        X_sub = self.X[self.clusters == k]
        
        Z = np.zeros((X_sub.shape[0],len(self.W[k].keys())))
                       
        for id, w in enumerate(self.W[k].keys()):
            w_concat = np.concatenate([np.clip(np.array(self.W[k][w]).T.reshape(-1),0,None), 
                                       -np.clip(-1*np.array(self.W[k][w]).T.reshape(-1),0,None)])
                       
            Z[:,id] = (X_sub @ w_concat + self.b[k][w] > 0).astype(np.int)
        
        return Z

    def generateYMatrix(self):
        Y = {}
        for cl in self.cluster_unique:
            Y_cl = np.zeros((int(self.X.shape[1]/2),len(self.W[cl].keys())))                                        
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
            self.alpha_constr = self.m.addConstr(self.weights @ self.eps <= self.alpha, name='alpha')
            self.alpha_constrs += 1
            
        #Add constraint to ensure we meet item demand
        for id,cl in enumerate(self.cluster_unique):
            Z = self.generateZMatrix(cl)
            self.m.addConstr(Z @ self.z[cl] + self.eps[self.clusters != cl] >= 1,name='set_cover_k%d'%cl)
            
        #Add constraint to ensure we meet item demand
        for id, cl in enumerate(self.cluster_unique):
            Z = self.generateZ2Matrix(cl)
            self.m.addConstr(- Z @ self.z[cl] + self.M*self.eps[self.clusters == cl] >= 0,name='set_cover_k_self%d'%cl)

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

        elif self.obj_type == 'sparsity':
            self.m.setObjective(self.y.sum(), gp.GRB.MINIMIZE)
            
        elif self.obj_type == 'feasibility':
            if self.bad_hps > 0:
                self.m.setObjective(self.X.shape[0]*self.z[:self.bad_hps].sum() + self.eps.sum(), gp.GRB.MINIMIZE)
            else:
                self.m.setObjective(self.weights @ self.eps, gp.GRB.MINIMIZE)
