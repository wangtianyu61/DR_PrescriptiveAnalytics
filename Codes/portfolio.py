import pandas as pd
import numpy as np
import math
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import *

class Portfolio:
    def __init__(self, df_select, window_size, feature_name, label_name, kernel_type, unit):
        #self.df_train = df_train
        self.all_covariate = np.array(df_select[feature_name])
        self.all_return = np.array(df_select[label_name])
        #window size controls the month for training
        self.window_size = window_size
        self.sample_number = len(df_select)
        
        #self.df_test = np.array(df_test)
        self.feature_name = feature_name
        self.portfolio_number = len(label_name)
        self.test_number = self.sample_number - self.window_size
        #optimization parameter
        ##finally obtained by cross validation
        self.ambiguity_size = 0.05*math.pow(self.window_size, 1/len(label_name))
        
        #model parameter (b and h)
        self.epsilon = 0.05
        self.tradeoff_param = 1
        self.kernel_type = kernel_type
        self.kernel_norm = 2
        #covariate information (no prior)
        self.covariate_weight = np.ones(self.sample_number)/self.sample_number
        self.unit = unit
        
    def optimize_kernel(self, robust):
        
        # variables
        m = Model("portfolio_robust_prescriptiveanalytics")
        lag_lambda = m.addVar(name = "lambda", lb = 0)
        ## no shortsale constraint (decision constraint)
        weight = pd.Series(m.addVars(self.portfolio_number, lb = 0))
        m.addConstr(weight.sum() == 1, 'budget')
        #used in the cvar formulation 
        v = m.addVar(name = 'adj_v', lb = -GRB.INFINITY)
        ## auxiliary variables
        aux_s = pd.Series(m.addVars(self.window_size, lb = -GRB.INFINITY))

        # constraint part

        ## nominal constraint
        #m.addConstrs((-self.lost_cost*order + self.history_demand[i]*aux_gamma[i][0] + (self.demand_upper_bound - self.history_demand[i])*aux_gamma[i][1] >=0 for i in range(self.sample_number)), "c0")
        m.addConstrs((-(1/self.epsilon + self.tradeoff_param)*np.dot(self.history_return[i], weight) + v*(1 - 1/self.epsilon) <= aux_s[i]
                      for i in range(self.window_size)), 'c0')        
        m.addConstrs((-self.tradeoff_param*np.dot(self.history_return[i], weight) + v <= aux_s[i]
                      for i in range(self.window_size)), 'c1')
        ## auxiliary constraint
        m.addConstrs(((1/self.epsilon + self.tradeoff_param)*weight[i]<= lag_lambda
                      for i in range(self.portfolio_number)), 'c10')
        m.addConstrs(((1/self.epsilon + self.tradeoff_param)*weight[i]>= -lag_lambda
                      for i in range(self.portfolio_number)), 'c11')        
        m.addConstrs((self.tradeoff_param*weight[i]<= lag_lambda
                      for i in range(self.portfolio_number)), 'c20')
        m.addConstrs((self.tradeoff_param*weight[i]>= -lag_lambda
                      for i in range(self.portfolio_number)), 'c21')
        
        #target for the worst-case CVaR
        if robust == 1:
            obj = self.ambiguity_size * lag_lambda
        else:
            obj = 0
        for i in range(self.window_size):
            obj += self.covariate_weight[i]*aux_s[i]
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        self.portfolio_weight = [v.x for v in weight]
       
    
    
    #evaluate the Sharpe Ratio / CVaR
    def strategy_evaluate(self):
        
        VaR = -np.percentile(self.portfolio_return, 100*self.epsilon)
        CVaR = np.zeros(self.test_number)
        count_CVaR = 0
        for i in range(self.test_number):
            if self.portfolio_return[i] < -VaR:
                count_CVaR += 1
                CVaR[i] = -self.portfolio_return[i]
        Sharpe_ratio = np.mean(self.portfolio_return)/np.std(self.portfolio_return, ddof = 1)
        
        CVaR = np.mean(CVaR)/(count_CVaR/self.test_number)
        CEQ = np.mean(self.portfolio_return/self.unit) - 1*np.std(self.portfolio_return/self.unit)**2
        print('Sharpe Ratio = ', Sharpe_ratio)
        print('Empirical CVaR = ', CVaR/self.unit)
        print('CEQ = ', CEQ)
        
        
    #weight_type: 'average', 'kernel'
    #robust: 0(none), 1(Wasserstein)
    #apply a rolling-based model for four strategies
    def evaluate(self, weight_type, robust, ambiguity_param = False):
        if type(ambiguity_param)!= bool:
            self.ambiguity_size = ambiguity_param*math.pow(self.window_size, 1/self.portfolio_number)
            print(self.ambiguity_size)
        #represent the portfolio return in our allocation model
        self.portfolio_return = np.zeros(self.test_number)
        #evaluate the average cost across the time
        for i in range(self.test_number):
            new_covariate = self.all_covariate[self.window_size + i]
            #rolling for the train-test-split
            self.history_covariate = self.all_covariate[i:(i + self.window_size)]
            self.history_return = self.all_return[i:(i + self.window_size)]
            self.test_return = self.all_return[self.window_size + i]
        
            #kernel estimate
            self.weight_estimate(new_covariate, weight_type)
            #optimize and derive the portfolio weight
            self.optimize_kernel(robust)
            self.portfolio_return[i] = np.dot(self.portfolio_weight, self.test_return)
        self.strategy_evaluate()
    
    #apply Equally Weighted Portfolio Method
    def evaluate_EW(self):
        self.portfolio_return = np.zeros(self.test_number)
        self.test_return = self.all_return[self.window_size:]
        self.portfolio_weight = np.ones(self.portfolio_number)/self.portfolio_number
        self.portfolio_return = np.dot(self.portfolio_weight, self.test_return.T)
        self.strategy_evaluate()
    #kernel choice
    def kernel(self, x):
        x_norm = np.linalg.norm(x, ord = 2)
        #print(x_norm)
        #infinite support
        if self.kernel_type == 'gaussian':
            #print('norm is ', x_norm)
            return math.exp(-(x_norm)**2)
        else:
            #naive finite support
            if self.kernel_type == 'naive':
                if x_norm < 0.25:
                    return 1
                else:
                    return 0
            #not trivial finite support
            
    #bandwidth choice (consider the convergence rate for the NW estimator)
    def bandwidth(self):
        #threshold in 
        bw_order = 1/(len(self.feature_name) + 2)
        return 1.0*self.sample_number*(-bw_order)
    
    def weight_estimate(self, covariate, weight_type):
        #update the covariate weight 
        #self.df_train means the history covariate
        if weight_type == 'average':
            self.covariate_weight = np.ones(self.sample_number)/self.sample_number
        else:
            for i in range(self.window_size):
                #print((covariate - self.history_covariate[i])/self.bandwidth())
            
                self.covariate_weight[i] = self.kernel((covariate - self.history_covariate[i])/(self.bandwidth()*0.1))
                #print(self.covariate_weight[i])
            self.covariate_weight /= np.sum(self.covariate_weight)
            
if __name__ == '__main__':
    feature_name = ['Mkt-RF', 'SMB', 'HML']
    unit = 100
    param_choice = [0.2, 0.4, 0.8]
    #['NoDur','Durbl','Manuf','Enrgy', 'HiTec', 'Telcm', 'Shops', 'Hlth' , 'Utils', 'Other']
    df = pd.read_csv("../PortfolioData/3_Factor_10_Industry.csv")
    rv_name = df.columns[4:]
    portfolio_model = Portfolio(df, 60, feature_name, rv_name, 'gaussian', unit)
    portfolio_model.evaluate_EW()
    portfolio_model.evaluate('kernel', 1, param_choice[0])
    portfolio_model.evaluate('kernel', 1, param_choice[1])
    portfolio_model.evaluate('kernel', 1, param_choice[2])
    portfolio_model.evaluate('kernel', 0)   
    
    portfolio_model.evaluate('average', 1, param_choice[0])
    portfolio_model.evaluate('average', 1, param_choice[1])
    portfolio_model.evaluate('average', 1, param_choice[2])
    portfolio_model.evaluate('average', 0)
    
    

