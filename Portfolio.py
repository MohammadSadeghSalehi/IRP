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

I = 7
B_PortTot = 150
Data = []
for i in range (I):
    Data.append([])
Budget = np.loadtxt('Budget.txt', usecols=range(I-1))
Data = np.loadtxt('Data.txt', usecols=range(I-1))
ExpectedGain = np.loadtxt('Expected.txt', usecols=range(I-1))
print(ExpectedGain)