import pandas as pd
import numpy as np
import math
import random
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from gurobipy import *
from matplotlib import rcParams

config = {
    "font.family":'serif',
    #"font.size":7.5,
    "mathtext.fontset":'stix',
    "font.serif":['SimHei'],    
}
rcParams.update(config)
exp_choice = 3

class Newsvendor2:
    def __init__(self, df_train, df_test, feature_name, label_name, kernel_type, ambiguity_set_type = False):
        self.df_train = df_train
        self.history_covariate = np.array(df_train[feature_name])
        self.test_covariate = np.array(df_test[feature_name])
        self.history_demand = np.array(list(df_train[label_name]))
        self.test_demand = np.array(list(df_test[label_name]))
        self.df_test = np.array(df_test)
        self.feature_name = feature_name
        self.sample_number = len(df_train)
        self.test_number = len(df_test)
        #optimization parameter
        ##chose the ambiguity set
        if type(ambiguity_set_type)!= bool:
            if ambiguity_set_type[0] == 0:
                self.ambiguity_size = ambiguity_set_type[1]/math.sqrt(self.sample_number)
            else:
                self.ambiguity_size = ambiguity_set_type[1]
        else:
            self.ambiguity_size = 25/math.sqrt(self.sample_number)
        #40/math.sqrt(self.sample_number)
        #upper bound of the stochastic demand
        self.demand_upper_bound = max(max(self.history_demand), max(self.test_demand)) + 1
        self.demand_lower_bound = min(min(self.history_demand)-1, min(self.test_demand)- 1, 0)
        #model parameter (b and h)
        self.lost_cost = 1
        self.out_cost = 10
        self.kernel_type = kernel_type
        self.kernel_norm = 2
        #covariate information (no prior)
        self.covariate_weight = np.ones(self.sample_number)/self.sample_number
    
    def optimize(self):
        
        # variables
        m = Model("newsvendor_robust_prescriptiveanalytics")
        lag_lambda = m.addVar(name = "lambda")
        order = m.addVar(name = 'order')
        aux_s = pd.Series(m.addVars(self.sample_number, lb = -GRB.INFINITY))
        aux_gamma = []
        for i in range(self.sample_number):
            aux_gamma.append(pd.Series(m.addVars(4)))
        # constraint part
        ## possible order constraint
        m.addConstr(order <= self.demand_upper_bound, 'c0')
        ## nominal constraint
        #m.addConstrs((-self.lost_cost*order + self.history_demand[i]*aux_gamma[i][0] + (self.demand_upper_bound - self.history_demand[i])*aux_gamma[i][1] >=0 for i in range(self.sample_number)), "c0")
        m.addConstrs((-self.lost_cost*order + (self.history_demand[i] - self.demand_lower_bound)*aux_gamma[i][0] + (self.demand_upper_bound - self.history_demand[i])*aux_gamma[i][1] 
                      <= aux_s[i] - self.lost_cost*self.history_demand[i] for i in range(self.sample_number)), "c1")
        m.addConstrs((self.out_cost*order + (self.history_demand[i] - self.demand_lower_bound)*aux_gamma[i][2] + (self.demand_upper_bound - self.history_demand[i])*aux_gamma[i][3] 
                      <= aux_s[i] + self.out_cost*self.history_demand[i] for i in range(self.sample_number)), "c2")
        ## auxiliary constraint
        m.addConstrs((-aux_gamma[i][0] + aux_gamma[i][1] >= -lag_lambda + self.lost_cost
                      for i in range(self.sample_number)), "c3")
        m.addConstrs((-aux_gamma[i][0] + aux_gamma[i][1] <= lag_lambda + self.lost_cost
                      for i in range(self.sample_number)), "c4")
        m.addConstrs((-aux_gamma[i][2] + aux_gamma[i][3] >= -lag_lambda - self.out_cost
                      for i in range(self.sample_number)), "c5")
        m.addConstrs((-aux_gamma[i][2] + aux_gamma[i][3] <= lag_lambda - self.out_cost
                      for i in range(self.sample_number)), "c6")
        #target for the demand
        obj = self.ambiguity_size * lag_lambda
        #obj = 0
        for i in range(self.sample_number):
            obj += self.covariate_weight[i]*aux_s[i]
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        self.order_value = order.x
        
        # #print the constraint output
        # print('lambda constraint = ', lag_lambda.x)
        # s_value = [aux_s[i].x for i in range(self.sample_number)]
        # print("s = ", s_value)
        # gamma_value = [[aux_gamma[i][0].x, aux_gamma[i][1].x, aux_gamma[i][2].x, aux_gamma[i][3].x]
        #                 for i in range(self.sample_number)]
        # print('gamma = ', gamma_value)
        return m.objVal
    
    def optimize_kernel(self):
        #variables
        m = Model("newsvendor_prescriptiveanalytics")
        order = m.addVar(name = 'order')
        aux_s1 = pd.Series(m.addVars(self.sample_number))
        aux_s2 = pd.Series(m.addVars(self.sample_number))
        # constraint part
        m.addConstr(order <= self.demand_upper_bound, 'c0')
        m.addConstrs((aux_s1[i] >= self.history_demand[i] - order
                        for i in range(self.sample_number)), 'c1')
        m.addConstrs((aux_s2[i] >= order - self.history_demand[i]
                          for i in range(self.sample_number)), 'c2')
        #target for the demand
        obj = 0
        for i in range(self.sample_number):
            obj += self.covariate_weight[i]*(self.lost_cost*aux_s1[i] + self.out_cost*aux_s2[i])
        m.setObjective(obj, GRB.MINIMIZE)
        m.setParam('OutputFlag', 0)
        m.optimize()
        self.order_value = order.x
        
        return m.objVal
    
    #sample generation mechanism
    def probability_compare(self, DRO_cost):
        sample_number = 50
        base_noise_std = 4
        weekend_sign = np.zeros(sample_number)
        temp = np.random.normal(20, 2, sample_number)
        for i in range(sample_number):
            if random.random() > 5/7:
                weekend_sign[i] = 1
        df = pd.DataFrame(pd.Series(temp))
        df['weekend'] = pd.Series(weekend_sign) 
        #df.columns = ['temp', 'weekend']
        #demand
        demand = np.zeros(sample_number)
        cost = np.zeros(sample_number)
        
        temp = self.new_covariate[0]
        weekend_sign = self.new_covariate[1]
        disappoint_num = 0
        for i in range(sample_number):
            base_demand = 100 + (temp - 20) + 20*weekend_sign        
            demand[i] = np.random.normal(base_demand, base_noise_std)
            cost[i] = max(self.lost_cost*(demand[i] - self.order_value), self.out_cost*(self.order_value - demand[i])) 
            if cost[i] > DRO_cost:
                disappoint_num += 1
        return disappoint_num / sample_number
    
    #evaluate the cost of newsvendor problem
    def cost_evaluate(self, decision_order, real_demand):
        
        return      
    
    #weight_type: 'average', 'kernel'
    #robust: 0(none), 1(Wasserstein)
    def evaluate_prob(self, weight_type, robust, ambiguity_set_type = False):
        if type(ambiguity_set_type)!= bool:
            if ambiguity_set_type[0] == 0:
                self.ambiguity_size = ambiguity_set_type[1]/math.sqrt(self.sample_number)
            else:
                self.ambiguity_size = ambiguity_set_type[1]
        disappoint_prob = 0
        true_cost = 0
        #evaluate the average cost across the time
        for i in range(self.test_number):
            self.new_covariate = self.test_covariate[i]
            #kernel estimate
            self.weight_estimate(self.new_covariate, weight_type)
            #optimize and derive
            if robust == 0:
                est_cost = self.optimize_kernel()
            else:
                if robust == 1:
                    est_cost = self.optimize()
            #print(order_x, self.test_demand[i])
            disappoint_prob += self.probability_compare(est_cost)
        return disappoint_prob / self.test_number

    #kernel choice
    def kernel(self, x):
        x[1] = 10*x[1]
        x_norm = np.linalg.norm(x, ord = 2)
        #print(x_norm)
        #infinite support
        if self.kernel_type == 'gaussian':
            #print('norm is ', x_norm)
            return math.exp(-(x_norm)**2)
        else:
            #naive finite support
            if self.kernel_type == 'naive':
                if x_norm < 1:
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
            for i in range(self.sample_number):
                #print((covariate - self.history_covariate[i])/self.bandwidth())
            
                self.covariate_weight[i] = self.kernel((covariate - self.history_covariate[i])/(self.bandwidth()*0.1))
                #print(self.covariate_weight[i])
            #print(self.history_demand, self.covariate_weight)
            self.covariate_weight /= np.sum(self.covariate_weight)
            

#alg_cost is a dictionary contains the four algorithms
def alg_compare_plot(sample_list, alg_cost, language):
    if len(alg_cost) == 4:
        marker_choice = ['*','*', 'o','o', '.', '.']
        linestyles = ['-',':','-',':','-',':']
    else:
        marker_choice = ['*', 'o','.', '*', 'o', '.']
        linestyles = ['-','-','-',':',':',':']
    sns.set_style('darkgrid')
    plt.figure(figsize = (10, 6), dpi = 100)
    #in chinese

    for i, model_name in enumerate(alg_cost.keys()):
        plt.plot(sample_list, alg_cost[model_name], label = model_name, linestyle = linestyles[i], 
                 marker = marker_choice[i], linewidth = 1)
    pyplot.xticks(sample_list)

    plt.xlabel('sample size')
    plt.ylabel('Out-of-Sample Disappointment')
    plt.legend(loc = 'upper right')
    plt.savefig('../output/nv' + str(exp_choice) + 'en.pdf')
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv('../output/nv3.csv')
    train_number_list = list(df['train_number'])
    alg_cost = {'Naive-SO':list(df['Naive-SO']), 'Naive-DRO':list(df['Naive-DRO']),
                'NW-SO':list(df['NW-SO']), 'NW-DRO':list(df['NW-DRO'])}
    alg_compare_plot(train_number_list, alg_cost, 'chinese')
    #pre_determined param for the ambiguity set size
    
    set_size1 = [40, 50, 60]
    set_size2 = [2, 4, 6]
    AS_type = [1, 2]
    train_number_list = [50*i for i in range(1, 11)]
    if exp_choice == 1 or exp_choice == 3:
        if exp_choice == 1:
            AS_type = [0, 40] #[20, 30, 50]
        else:
            AS_type = [1, 2] #[0.5, 1, 2]
        res_DRO_kernel = np.zeros(len(train_number_list))
        res_SO_kernel = np.zeros(len(train_number_list))
        res_DRO_naive = np.zeros(len(train_number_list))
        res_SO_naive = np.zeros(len(train_number_list))
    else:
        res_kernel1 = np.zeros(len(train_number_list))
        res_kernel2 = np.zeros(len(train_number_list))
        res_kernel3 = np.zeros(len(train_number_list))
        res_naive1 = np.zeros(len(train_number_list))
        res_naive2 = np.zeros(len(train_number_list))
        res_naive3 = np.zeros(len(train_number_list))   
             
    for j, train_number in enumerate(train_number_list):
        print(train_number)
        test_size = 50
        sample_number = train_number + test_size
        
        base_noise_std = 4
        weekend_sign = np.zeros(sample_number)
        temp = np.random.normal(20, 2, sample_number)
        for i in range(sample_number):
            if random.random() > 5/7:
                weekend_sign[i] = 1
        df = pd.DataFrame(pd.Series(temp))
        df['weekend'] = pd.Series(weekend_sign) 
        #df.columns = ['temp', 'weekend']
        #demand
        demand = np.zeros(sample_number)
        for i in range(sample_number):
            base_demand = 100 + (temp[i] - 20) + 20*weekend_sign[i]        
            demand[i] = np.random.normal(base_demand, base_noise_std)
        df['demand'] = pd.Series(demand)
        df.columns = ['temp', 'weekend', 'demand']
        
        df_train = df.loc[0:(train_number - 1)]
        df_test = df.loc[train_number:]
        
        feature_name = ['temp', 'weekend']
        label_name = 'demand'
        if exp_choice == 1 or exp_choice == 3:
            nv_model = Newsvendor2(df_train, df_test, feature_name, label_name, 'gaussian', AS_type)
            
            res_DRO_kernel[j] = nv_model.evaluate_prob('kernel', 1)
            res_DRO_naive[j] = nv_model.evaluate_prob('average', 1)
            res_SO_kernel[j] = nv_model.evaluate_prob('kernel', 0)
            res_SO_naive[j] = nv_model.evaluate_prob('average', 0)
        else:
            nv_model = Newsvendor2(df_train, df_test, feature_name, label_name, 'gaussian')
            if exp_choice == 2:
            
                res_kernel1[j] = nv_model.evaluate_prob('kernel', 1, [0, set_size1[0]])
                res_kernel2[j] = nv_model.evaluate_prob('kernel', 1, [0, set_size1[1]])
                res_kernel3[j] = nv_model.evaluate_prob('kernel', 1, [0, set_size1[2]])
                res_naive1[j] = nv_model.evaluate_prob('average', 1, [0, set_size1[0]])
                res_naive2[j] = nv_model.evaluate_prob('average', 1, [0, set_size1[1]])
                res_naive3[j] = nv_model.evaluate_prob('average', 1, [0, set_size1[2]])    
            else:
                print("/")
                #fix ambiguity set size
                res_kernel1[j] = nv_model.evaluate_prob('kernel', 1, [1, set_size2[0]])
                res_kernel2[j] = nv_model.evaluate_prob('kernel', 1, [1, set_size2[1]])
                res_kernel3[j] = nv_model.evaluate_prob('kernel', 1, [1, set_size2[2]])
                res_naive1[j] = nv_model.evaluate_prob('average', 1, [1, set_size2[0]])
                res_naive2[j] = nv_model.evaluate_prob('average', 1, [1, set_size2[1]])
                res_naive3[j] = nv_model.evaluate_prob('average', 1, [1, set_size2[2]])                     
    if exp_choice in [1, 3]:
        alg_cost = {'NW-DRO': res_DRO_kernel, 'NW-SO': res_SO_kernel, 'Naive-DRO': res_DRO_naive, 'Naive-SO': res_SO_naive}
    else:
        if exp_choice == 2:
            alg_cost = {'NW-DRO, C = '+ str(set_size1[0]): res_kernel1,
                        'NW-DRO, C = '+ str(set_size1[1]): res_kernel2,
                        'NW-DRO, C = '+ str(set_size1[2]): res_kernel3,
                        'Naive-DRO, C = '+ str(set_size1[0]): res_naive1,
                        'Naive-DRO, C = '+ str(set_size1[1]): res_naive2,
                        'Naive-DRO, C = '+ str(set_size1[2]): res_naive3}
        else:
            alg_cost = {r'NW-DRO, $\varepsilon$ = '+ str(set_size2[0]): res_kernel1,
                        r'NW-DRO, $\varepsilon$ = '+ str(set_size2[1]): res_kernel2,
                        r'NW-DRO, $\varepsilon$ = '+ str(set_size2[2]): res_kernel3,
                        r'Naive-DRO, $\varepsilon$ = '+ str(set_size2[0]): res_naive1,
                        r'Naive-DRO, $\varepsilon$ = '+ str(set_size2[1]): res_naive2,
                        r'Naive-DRO, $\varepsilon$ = '+ str(set_size2[2]): res_naive3}
    alg_compare_plot(train_number_list, alg_cost, 'english')

    