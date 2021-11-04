import numpy as np
# Portfolio Defenition list
# I Number of drugs in the portfolio.
# αi Type I error rate for drug i.
# Ji Number of possible designs for drug i.
# µ1i Value of true treatment effect θi when drug i is efficacious.
# µ0i Value of true treatment effect θi when drug i is not efficacious.
# σi2 Response variance for drug i.
# βi,j Type II error rate for drug i with design j when θi = µ1i .
# ni,j Total sample size required for each Phase III trial with drug i, design j. Deduced
#      from other parameters such as βi,j , αi , and σi2 .
# bi,j Budget required for each Phase III trial with design j on drug i ($M).
# P oSTi,j The probability of success (rejection of the null hypothesis) for each Phase III
#          trial with design j for drug i.
# n_i^trials Number of successful trials required for Phase III for drug i to market the drug.
# P oSi,j The probability of success of all n_i^trials trials for drug i.

# ei,j Expected gain from drug i with design j ($M).
# B_PortTot Total portfolio budget.
# θi True treatment effect for drug i.
# t_i^(a) Time of availability of drug i.
# p_i^(a) Probability of availability of drug i.
# p_i^eff Probability that drug i is efficacious.

# Dynamic Programming (Design History) Definition List
#Compute Optimal decision
# Data initialization Step 1
I = 7
B_PortTot = 150
N_design = 6
Budget = np.loadtxt('Budget.txt', usecols=range(I-1))
Data = np.loadtxt('Data.txt', usecols=range(I))
ExpectedGain = np.loadtxt('Expected.txt',usecols=range(I-1))

# Compute Designs Step 2

J = []
J.append(I)
J.append(B_PortTot)
for i in range(I-1):
    J_i = []
    J_i.append(Data[1,i])  #P_i^a 
    J_i.append(Budget[i,:]) #b_ij
    J_i.append(ExpectedGain[i,:]) #e_ij
    J_i.append(Data[16,i]) #n_i^trials
    J.append(J_i)
P_ia = Data[1,:]
#initial computations Step 3


def Master_Fun(I):
    eG_Arr =[]
    jStar = []
    for i in range(I-1,-1,-1):
        eG_Arr ,jStar = Update_Opt_Decs(i, eG_Arr, jStar)
    return eG_Arr,jStar

def eps_i(i,B):
    Jcal = []
    for j in range(N_design):
        temp = sum(Budget[:,j])- Budget[i,j]
        temp = B_PortTot - temp
        for k in range(N_design):
            if temp >= Budget[i,k]:
                Jcal.append(k)

def Update_Opt_Decs(i, eG_Arr, jStar):
    Jcal = []
    for j in range(N_design):
        temp = sum(Budget[:,j])- Budget[i,j]
        temp = B_PortTot - temp
        for k in range(N_design):
            if temp >= Budget[i,k]:
                Jcal.append(k)
    if i == I-1:
        temp = []
        for index in Jcal:
            temp.append(ExpectedGain[i,index])
        epsilon_i = P_ia[i]*np.amax(temp, axis=0)
        jStar.append(np.argmax(temp)) 
        eG_Arr.append(epsilon_i)
    else:
        temp = []
        for index in Jcal:
            temp.append(ExpectedGain[i,index]+eG_Arr[-1])
        epsilon_i = P_ia[i]*np.amax(temp) + (1-P_ia[i])*eG_Arr[-1]  #ExpectedGain[i+1,0]
        eG_Arr.append(epsilon_i)
        jStar.append(np.argmax(temp))
    return eG_Arr,jStar

eG , Js = Master_Fun(I)
print(eG)
print(Js)