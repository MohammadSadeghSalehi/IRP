import numpy as np
import matplotlib.pyplot as plt
from numpy import ma
from ortools.linear_solver import pywraplp
import sys
import time
np.set_printoptions(threshold = sys.maxsize)
np.random.seed(0)
#Objective function is max (ENPV-variance(NPV)) but as I did not have NPV here it is just max(ENPV)
#NPV is r_i and e_i is the expected gain
#Constraitns are decision variables and budget

def init(B,I,N_design):

    #Data initialization

    Budget = np.loadtxt('Budget.txt', usecols=range(N_design))
    Data = np.loadtxt('Data.txt', usecols=range(I))
    ExpectedGain = np.loadtxt('Expected.txt',usecols=range(N_design))
    varNPV = np.zeros((7,6))
    Pos = np.zeros((I,N_design))
    one_beta = Data[18:24,:]
    P_i_a = Data[1,:]
    alpha_i = Data[3,:]
    n__trials = Data[16,:]
    r = np.zeros((I,N_design))

    #calculating PoSij
    for i in range(I):
        for j in range(N_design):
            Pos[i][j] = (one_beta[j][i]**n__trials[i])*P_i_a[i] + (alpha_i[i]**n__trials[i])*(1-P_i_a[i])
            if Pos[i][j] != 0: r[i][j] = (ExpectedGain[i][j]+Budget[i][j]*(1-Pos[i][j]))/Pos[i][j]
            else:r[i][j] = 0
    #calculating varNPV        
    for i in range(I):
        for j in range(N_design):
            varNPV[i][j] = np.sum((ExpectedGain-r[i][j])**2)/(I*N_design-1)

    #lambda = max(e_ij)/max(varNPV)
    lmbda = np.max(ExpectedGain)/np.max(varNPV)
    
    return Budget, varNPV, ExpectedGain, lmbda

    
def __main__(B,printFlag):
    I = 7
    B_Total = B
    N_design = 6
    Budget,varNPV,ExpectedGain,lmbda = init(B,I,N_design)

    # Create the mip solver with the SCIP backend.
    # Also CBC can be used
    solver = pywraplp.Solver.CreateSolver('CBC')
    x = {}
    for i in range(I):
        for j in range(N_design):
            x[(i, j)] = solver.IntVar(0, 1, 'x_%i_%i' % (i, j))
            

    # Constraints
    # Each drug can have at most one selected design.
    for i in range(I):
        solver.Add(sum(x[i, j] for j in range(N_design)) <= 1)
    # Budget.
    solver.Add(
        sum(x[(i, j)] * Budget[i][j] for i in range(I)
        for j in range(N_design)) <= B_Total)

    # Objective
    objective = solver.Objective()

    for i in range(I):
        for j in range(N_design):
            objective.SetCoefficient(x[(i, j)], ExpectedGain[i][j]-lmbda*varNPV[i][j])
    objective.SetMaximization()
###################
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        if printFlag : print('Objective function value', objective.Value())
        total_cost = 0
        total_gain = 0
        for i in range(I):
            cost = 0
            eGain = 0
            if printFlag :print('Drug ', i+1, '\n')
            for j in range(N_design):
                if x[i, j].solution_value() > 0:
                    if printFlag :print('Design', j+1, '- Cost:', Budget[i][j], ' Expected gain:',
                        ExpectedGain[i][j])
                    cost += Budget[i][j]
                    eGain += ExpectedGain[i][j]
            if printFlag :print('Drug budget:', cost)
            if printFlag :print('Drug gain:', eGain)
            if printFlag :print()
            total_cost += cost
            total_gain += eGain
        if printFlag :print('Total spent money:', total_cost)
        if printFlag :print('Total gained money:', total_gain)

        return total_gain
    else:
        if printFlag :print('The problem does not have an optimal solution.')
        else: return 0



##Experiment
bInterval = np.linspace(0,300,num=100)
expectedReturn=[]
t = time.time()
for b in bInterval:

    expectedReturn.append(__main__(b,False))

##The Optimal Decision for B = 150 M$ 
__main__(150,True)
t=time.time()-t
print(t)





##Plot

plt.plot(bInterval,expectedReturn)
plt.xlabel('Total Budget ($M)')
plt.ylabel('Expexted Gain ($M)')
plt.show()