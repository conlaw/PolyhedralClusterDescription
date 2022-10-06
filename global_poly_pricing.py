import pandas as pd
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB


#Methods to give back dual values in relax mode

class GlobalPolyClusterExplanationPP01(object):
    
    def __init__(self, X, clusters, k, M = 10, beta = 1, objective = 'num_hp'):
        
        #Save constants
        #self.k = cluster_to_explain
        self.X = X.values
        self.k = k
        self.clusters = clusters
        self.clusters_unique = np.unique(clusters)
        #self.X_pos = self.X[self.clusters == self.k].values
        #self.X_neg = self.X[self.clusters != self.k].values
        self.epsilon = 0.001 #update smarter
        self.M = M
        self.beta = beta
        pricing_consts = {
            'sparsity': 0,
            'num_hp': 1,
            'feasibility': 0,
            'complex': 1
        }
        self.objective = objective
        self.pricing_const = pricing_consts[objective]
        
        #Initialize model
        self.init_model()
        
    
    def init_model(self):
        #Create gurobi model object
        self.m = gp.Model("cluster_explanation")
        
        #Initialize model's variables, consraints and objectives
        self.init_dv()
        self.add_constr()
        #self.set_obj()
        
        #Update the model to save all the changes
        self.m.update()
        
    def solve(self, mu_self, mu_prime, gamma, verbose = False):
        self.mu_self = mu_self
        self.mu_prime = mu_prime

        self.gamma = gamma
        self.set_obj()
        
        self.m.write('pricing.lp')
        # Set time limit
        self.m.Params.TimeLimit = 300
        
        #Set how chatty we want our results to be
        self.m.Params.OutputFlag = verbose
                
        self.m.optimize()
        
        W, B = self.getAllHPs()
        #Return various metrics about the run
        #'w': np.array([np.array([[dv.x - dv2.x for dv, dv2 in zip(self.w_pos, self.w_neg)]]).reshape(-1)]),
        # 'b': np.array([self.b.x]),

        return {
            'num_constr': self.m.NumConstrs,
            'num_vars': self.m.NumVars,
            'gap': self.m.MIPGap,
            'b&b_nodes': self.m.NodeCount,
            'time': self.m.Runtime,
            'm' : self.m,
            'w': W,
            'b': B,
            'newHP': self.pricing_const - self.m.objVal < 0,
            'objVal':self.m.objVal
        }
        
    def init_dv(self):
        #Create w, b vars
        self.w_pos = self.m.addMVar(self.X.shape[1], lb = 0, ub = self.M, vtype = gp.GRB.INTEGER, name='w_pos')
        self.w_neg = self.m.addMVar(self.X.shape[1], lb = 0, ub = self.M, vtype = gp.GRB.INTEGER, name='w_neg')
        self.b = self.m.addMVar(1, lb = -self.beta*np.max(np.abs(self.X))*self.M, ub = self.beta*np.max(np.abs(self.X))*self.M, vtype = gp.GRB.CONTINUOUS, name='b')
        
        #create y vars
        self.y_pos = self.m.addMVar(self.X.shape[1], vtype = gp.GRB.BINARY, name='y_pos')
        self.y_neg = self.m.addMVar(self.X.shape[1], vtype = gp.GRB.BINARY, name='y_neg')
        
        #delta i
        self.delta_i = self.m.addMVar(self.X.shape[0], vtype = gp.GRB.BINARY, name='delta_i')

    def add_constr(self):
        #Add constraints to remove trivial solution
        self.sparsity = self.m.addConstr(self.y_pos.sum() + self.y_neg.sum() <= self.beta, name='beta')
        self.y_nondegen = self.m.addConstr(self.y_pos + self.y_neg <= np.ones(self.X.shape[1]), name='y_nondegen')
        self.w_nondegen = self.m.addConstr(self.w_pos.sum() + self.w_neg.sum() >= 1, name='w_nondegen')

        self.w_pos_constrs = self.m.addConstr(self.w_pos <= self.M*self.y_pos, name='w_pos')
        self.w_neg_constrs = self.m.addConstr(self.w_neg <= self.M*self.y_neg, name='w_neg')
        
        bigM = 2*self.beta*np.max(np.abs(self.X))*self.M
        
        #delta i consts
        X_cl = self.X[self.clusters == self.k]
        X_sub = self.X[self.clusters != self.k]

        self.m.addConstr(X_sub @ self.w_pos - X_sub @ self.w_neg + np.ones((X_sub.shape[0],1))@self.b   >= self.epsilon-bigM+bigM*self.delta_i[self.clusters != self.k] , name='hp_sub')
        self.m.addConstr(X_cl @ self.w_pos - X_cl @ self.w_neg + np.ones((X_cl.shape[0],1))@self.b  <= self.epsilon+bigM*self.delta_i[self.clusters == self.k] , name='hp_cl')


    def set_obj(self):
        #Minimize # of hps used
        #print(self.mu_self, self.mu_prime)
        
        #Adjustment for having complexity in objective of pricing
        if self.objective == 'complex':
            y_add = 1
        else:
            y_add = 0
            
        self.m.setObjective( - self.mu_self @ self.delta_i[self.clusters == self.k] + self.mu_prime @ self.delta_i[self.clusters != self.k] + (self.gamma - y_add) @ self.y_pos + (self.gamma - y_add)@ self.y_neg , gp.GRB.MAXIMIZE)
    
    def getAllHPs(self):
        '''
        Return all rules with negative reduced costs
        '''
        
        solCount = self.m.SolCount
        rules = []
        W = {}
        B = {}
        #Loop through stored solutions and keep if negative reduced cost
        for i in range(solCount):
            self.m.Params.SolutionNumber = i
            
            obj = self.m.getAttr(GRB.Attr.PoolObjVal)
            if self.pricing_const - obj >= 0:
                break
            W[i] = np.array([np.array(self.m.getAttr(GRB.Attr.Xn)[0:self.X.shape[1]]) -np.array(self.m.getAttr(GRB.Attr.Xn)[self.X.shape[1]:2*self.X.shape[1]])])[0]
            B[i] = np.array([self.m.getAttr(GRB.Attr.Xn)[2*self.X.shape[1]]])
        
        
        return W,B
