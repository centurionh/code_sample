import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import random
import numpy as np
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factor.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()

    BayesNet.add_node('alarm')
    BayesNet.add_node('faulty alarm')
    BayesNet.add_node('gauge')
    BayesNet.add_node('faulty gauge')
    BayesNet.add_node('temperature')

    BayesNet.add_edge('gauge', 'alarm')
    BayesNet.add_edge('faulty alarm', 'alarm')
    BayesNet.add_edge('faulty gauge', 'gauge')
    BayesNet.add_edge('temperature', 'gauge')
    BayesNet.add_edge('temperature', 'faulty gauge')

    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """


    cpd_gauge = TabularCPD('gauge', 2, values=[[0.95,0.2,0.05,0.8], [0.05,0.8,0.95,0.2]], evidence=['temperature', 'faulty gauge'], evidence_card=[2,2])
    cpd_faulty_alarm = TabularCPD('faulty alarm', 2, values=[[0.85], [0.15]])
    cpd_temperature = TabularCPD('temperature', 2, values=[[0.8], [0.2]])
    cpd_faulty_gauge = TabularCPD('faulty gauge', 2, values=[[0.95, 0.2], [0.05, 0.8]], evidence=['temperature'], evidence_card=[2])   
    cpd_alarm = TabularCPD('alarm', 2, values=[[0.9, 0.55, 0.1, 0.45], [0.1, 0.45, 0.9, 0.55]], evidence=['gauge', 'faulty alarm'], evidence_card=[2,2])

    bayes_net.add_cpds(cpd_gauge, cpd_faulty_alarm, cpd_temperature, cpd_faulty_gauge, cpd_alarm)

    return bayes_net




def get_alarm_prob(bayes_net): 
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""

    solver = VariableElimination(bayes_net)

    marginal_prob = solver.query(variables=['alarm'])
    alarm_prob = marginal_prob['alarm'].values

    return alarm_prob[1]


def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TOOD: finish this function

    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'])
    gauge_prob = marginal_prob['gauge'].values
    return gauge_prob[1]


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['temperature'], evidence={'alarm':1, 'faulty alarm':0, 'faulty gauge':0})
    temp_prob = marginal_prob['temperature'].values
    return temp_prob[1]


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()

    BayesNet.add_node('A')
    BayesNet.add_node('B')
    BayesNet.add_node('C')
    BayesNet.add_node('AvB')
    BayesNet.add_node('BvC')
    BayesNet.add_node('CvA')

    BayesNet.add_edge('A','AvB')
    BayesNet.add_edge('A','CvA')
    BayesNet.add_edge('B','AvB')
    BayesNet.add_edge('B','BvC')
    BayesNet.add_edge('C','CvA')
    BayesNet.add_edge('C','BvC')

    cpd_A = TabularCPD('A', 4, values=[[0.15], [0.45], [0.3], [0.1]])
    cpd_B = TabularCPD('B', 4, values=[[0.15], [0.45], [0.3], [0.1]])
    cpd_C = TabularCPD('C', 4, values=[[0.15], [0.45], [0.3], [0.1]])
    
    prob_matrix = [[0.1, 0.2, 0.15, 0.05, 0.6, 0.1, 0.2, 0.15, 0.75, 0.6, 0.1, 0.2, 0.9,  0.75, 0.6, 0.1],
                   [0.1, 0.6, 0.75, 0.9,  0.2, 0.1, 0.6, 0.75, 0.15, 0.2, 0.1, 0.6, 0.05, 0.15, 0.2, 0.1],
                   [0.8, 0.2, 0.1,  0.05, 0.2, 0.8, 0.2, 0.1,  0.1,  0.2, 0.8, 0.2, 0.05, 0.1,  0.2, 0.8]]
    cpd_AvB = TabularCPD('AvB', 3, values=prob_matrix, evidence=['A','B'], evidence_card=[4,4])
    cpd_BvC = TabularCPD('BvC', 3, values=prob_matrix, evidence=['B','C'], evidence_card=[4,4])
    cpd_CvA = TabularCPD('CvA', 3, values=prob_matrix, evidence=['C','A'], evidence_card=[4,4])

    BayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_AvB, cpd_BvC, cpd_CvA)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    
    solver = VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['BvC'], evidence={'AvB':0, 'CvA':2})
    posterior = marginal_prob['BvC'].values
    return posterior


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values

    if not initial_state:
        initial_state = [random.randint(0,3) for i in range(3)] + [random.randint(0,2) for i in range(3)]
    sample = tuple(initial_state)

    # print('match_table ', match_table)
    # print('team_table ', team_table)    
    
    key = random.choice([0,1,2,3,4,5])
    distribution = []
    if key<=2:
        for i in range(4):
            initial_state[key] = i
            # print('initial_state ',initial_state)
            distribution.append(team_table[initial_state[0]] *\
                                team_table[initial_state[1]] *\
                                team_table[initial_state[2]] *\
                                match_table[initial_state[3]][initial_state[0]][initial_state[1]] *\
                                match_table[initial_state[4]][initial_state[1]][initial_state[2]] *\
                                match_table[initial_state[5]][initial_state[2]][initial_state[0]])
    else:
        for i in range(3):
            initial_state[key] = i
            # print('initial_state ',initial_state)
            distribution.append(team_table[initial_state[0]] *\
                                team_table[initial_state[1]] *\
                                team_table[initial_state[2]] *\
                                match_table[initial_state[3]][initial_state[0]][initial_state[1]] *\
                                match_table[initial_state[4]][initial_state[1]][initial_state[2]] *\
                                match_table[initial_state[5]][initial_state[2]][initial_state[0]])

    cut = random.random()
    total = 0
    for i in range(len(distribution)):
        total += distribution[i] / sum(distribution)
        if total > cut:
            initial_state[key] = i
            break

    sample = tuple(initial_state)   
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values

    if not initial_state:
        initial_state = [random.randint(0,3) for i in range(3)] + [random.randint(0,2) for i in range(3)]
    sample = tuple(initial_state)    
    
    prob = team_table[initial_state[0]] *\
           team_table[initial_state[1]] *\
           team_table[initial_state[2]] *\
           match_table[initial_state[3]][initial_state[0]][initial_state[1]] *\
           match_table[initial_state[4]][initial_state[1]][initial_state[2]] *\
           match_table[initial_state[5]][initial_state[2]][initial_state[0]]

    candidate = [random.randint(0,3) for i in range(3)] + [random.randint(0,2) for i in range(3)]
    candidate_prob = team_table[candidate[0]] *\
                team_table[candidate[1]] *\
                team_table[candidate[2]] *\
                match_table[candidate[3]][candidate[0]][candidate[1]] *\
                match_table[candidate[4]][candidate[1]][candidate[2]] *\
                match_table[candidate[5]][candidate[2]][candidate[0]]

    if prob < candidate_prob:
        sample = tuple(candidate)
        return sample
    else:
        cut = random.random()
        if cut < candidate_prob / prob:
            sample = tuple(candidate)
            return sample

    sample = tuple(initial_state)
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    
    burnin = 10000
    # Gibbs
    for i in range(burnin):
        sample = Gibbs_sampler(bayes_net, list(initial_state))
        Gibbs_convergence[sample[4]] += 1
    Gibbs_count = burnin
    convergence = [g / Gibbs_count for g in Gibbs_convergence]
    count = 0

    while 1:
        sample = Gibbs_sampler(bayes_net, list(sample))
        Gibbs_convergence[sample[4]] += 1
        Gibbs_count += 1

        update_convergence = [g / Gibbs_count for g in Gibbs_convergence]
        convergence = update_convergence
        
        delta = np.abs(np.array(convergence) - np.array(update_convergence))
        if np.all(delta < 0.001):
            count += 1
            if count >= 10 and Gibbs_count > burnin + 20000:
                break
        else:
            count = 0
    Gibbs_convergence = convergence

    # MH
    sample = initial_state
    for i in range(burnin):
        if sample == MH_sampler(bayes_net, sample):
            MH_rejection_count += 1
        sample = MH_sampler(bayes_net, sample)
        MH_convergence[sample[4]] += 1
    MH_count = burnin
    convergence = [m / 30000 for m in MH_convergence]
    count = 0 

    while 1:
        if sample == MH_sampler(bayes_net, sample):
            MH_rejection_count += 1
        sample = MH_sampler(bayes_net, sample)
        MH_convergence[sample[4]] += 1  
        MH_count += 1

        update_convergence = [m / MH_count for m in MH_convergence]
        convergence = update_convergence

        delta = np.abs(np.array(convergence) - np.array(update_convergence))
        if np.all(delta < 0.001):
            count += 1
            if count >= 10 and MH_count > burnin + 20000:
                break
        else:
            count = 0

    MH_convergence = convergence

    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 1
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    return 'Zhengyang He'
