# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:50:54 2020

@author: Cory Kromer-Edwards

Antibiotic Genetic Algorithm 
"""

from random import random
import numpy as np 
from functools import partial
import re

# import ml

from utils.antibiotic import Antibiotic
from utils import ModSMI

from deap import base
from deap import creator
from deap import tools

FASTA_PARSE_REGEX = re.compile('\>.+?\n([\w\n]+)')

# Set up for numpy warnings within the fitness evaluation methods
# By default, warnings are just printed to stderr rather than thrown
# We want warnings to be thrown as warnings to be able to catch them later.
# np.seterr(all='warn')
# import warnings

# =====================================================================================
# tools: A set of functions to be called during genetic algorithm operations (mutate, mate, select, etc)
# Documentation: https://deap.readthedocs.io/en/master/api/tools.html
# =====================================================================================

# =====================================================================================
# Creator: Creates a class based off of a given class (known as containers)
#   creator.create(class name, base class to inherit from, args**)
#     class name: what the name of the class should be
#     base class: What the created class should inherit from
#     args**: key-value argument pairs that the class should have as fields
#
#   EX: creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     Creates a class named "FitnessMax" that inherits from base fitness class
#     that library has (maximizes fitness value). It then has a tuple of weights
#     that are given as a field for the class to use later.
# =====================================================================================

# The base.Fitness function will try to maximize fitness*weight.
creator.create("AntFitnessMin", base.Fitness, weights=(-1.0, -0.5, -1.0, 0.5))
creator.create("AntIndividual", Antibiotic, fitness=creator.AntFitnessMin, generation=0)


# NOTE ON FITNESS WEIGHTS:
#   Weights will be used when finding the maximum fitness within the Deap library,
#   but you will see the fitness value that is return from evaluation function
#   IE: During fitness max function -> fitness * weights
#       When calling "individual.fitness.values" -> fitness / weights


# =====================================================================================
# Toolbox: Used to add aliases and fixed arguements for functions that we will use later.
#   toolbox.[un]register(alias name, function, args*)
#     alias name: name to give the function being added
#     function: the function that is being aliased in the toolbox
#     args*: arguments to fix for the function when calling it later
#
#   EX: toolbox.register("attr_bool", random.randint, 0, 1)
#     Creates an alias for the random.randint function with the name "attr_bool"
#       with the default min and max int values being passed in being 0 and 1.
#
#   EX: toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
#     Creates an alias for the tools.initRepeat function with the name "individual".
#     This function takes in the class that we want to repeadidly intialize from, the function to 
#     initialize values with, and how many values to create from that function. This will create
#     an individual with 100 random boolean values.
# =====================================================================================


def combine_antibiotics(antibiotic1, antibiotic2, prob_keep_ring):
  smiles1 = antibiotic1.get_smiles()
  smiles2 = antibiotic2.get_smiles()

  history1 = antibiotic1.get_history()
  history2 = antibiotic2.get_history()

  avoid_ring = random() <= prob_keep_ring
  new_smiles1, mol1 = ModSMI.crossover_smiles(smiles1, smiles2, avoid_ring)
  if mol1:
    new_history1 = [(history1, history2)]
  else:
    new_history1 = history1
    new_smiles1 = smiles1


  avoid_ring = random() <= prob_keep_ring
  new_smiles2, mol2 = ModSMI.crossover_smiles(smiles2, smiles1, avoid_ring)
  if mol2:
    new_history2 = [(history2, history1)]
  else:
    new_history2 = history2
    new_smiles2 = smiles2

  return Antibiotic(new_smiles1, history=new_history1), Antibiotic(new_smiles2, history=new_history2)


def mutate_antibiotic(antibiotic, add_prob, replace_prob, delete_prob):
  return antibiotic.mutate(add_prob, replace_prob, delete_prob),


class AntGenAlg():
  def __init__(self, smiles, add_prob=33, replace_prob=33, delete_prob=33, prob_keep_ring=0.5, num_gen=50, pop_size=100, tournsize=4, cxpb=0.5, debug=0):
    self.num_gen = num_gen
    self.pop_size = pop_size
    self.cxpb = cxpb
    self.debug = debug

    self.toolbox = base.Toolbox()

    self.stats = tools.Statistics(key=self._stat_func)
    self.stats.register("avg", np.mean)
    self.stats.register("std", np.std)
    self.stats.register("min", np.min)
    self.stats.register("max", np.max)

    # Set up ways to define individuals in the population
    # self.toolbox.register("attr_x", np.random.normal, 0, 2)
    self.toolbox.register("individual", creator.AntIndividual, smiles)
    self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

    # Set up ways to change population
    self.toolbox.register("mate", combine_antibiotics, prob_keep_ring=prob_keep_ring)
    self.toolbox.register("mutate", mutate_antibiotic, add_prob=add_prob, replace_prob=replace_prob, delete_prob=delete_prob)
    self.toolbox.register("select", tools.selTournament, tournsize=tournsize)

  def _stat_func(self, ind):
    # log_mic, mic_in_range = ml.safe_pred_to_log_mic(ind.fitness.values[0])
    log_mic, mic_in_range = -1, True
    if not mic_in_range:
      return 1000
    else:
      return log_mic


  def _print_fitnesses(self, g, fitnesses):
    predicted_mics = [fit[0] for fit in fitnesses]
    predicted_confidences = [fit[1] for fit in fitnesses]
    second_max_mics = [fit[2] for fit in fitnesses]
    next_mic_confidences = [fit[3] for fit in fitnesses]
    max_mic_index = np.argmax(predicted_mics)
    min_mic_index = np.argmin(predicted_mics)
    print(f"(min, max) Generation {g}\t \
      predicted MIC Index: ({predicted_mics[min_mic_index]}, {predicted_mics[max_mic_index]})\t \
      confidence: ({predicted_confidences[min_mic_index]:.3f}, {predicted_confidences[max_mic_index]:.3f})\t \
      next MIC Index: ({second_max_mics[min_mic_index]}, {second_max_mics[max_mic_index]})\t \
      next MIC confidence: ({next_mic_confidences[min_mic_index]:.3f}, {next_mic_confidences[max_mic_index]:.3f})")

  def evaluate_fitness(self, antibiotic_smiles, best_isolate_representation, model):
    # Classify antibiotic and smiles to get MIC (use that for base fitness)
    # predicted_mic_index, predicted_mic_confidence, next_min_mic_index, next_min_mic_confidence, smiles_exceeds_embeding = model.predict_for_smiles(best_isolate_representation, antibiotic_smiles)
    predicted_mic_index, predicted_mic_confidence, next_min_mic_index, next_min_mic_confidence, smiles_exceeds_embeding = 3, 0.87, 4, 0.76, False

    if smiles_exceeds_embeding:
      return (10 * predicted_mic_index, 10 * predicted_mic_confidence, 10 * next_min_mic_index, -1 * next_min_mic_confidence)
    else:
      return (predicted_mic_index, predicted_mic_confidence, next_min_mic_index, next_min_mic_confidence)

  def run(self, best_isolate_representation, model):
    """
    Run a genetic algorithm with the given evaluation function and input parameters.
    Main portion of code for this method found from Deap example at URL:
    https://deap.readthedocs.io/en/master/overview.html
  
    Parameters
    ----------
    None
  
    Returns
    -------
    best_individual: List
      The best individual found out of all iterations
    fitness: Float
      The best_individual's fitness value
    logbook : Dictionary
      A dictionary of arrays for iterations, min, max, average, and std. dev. for each iteration.
  
    """
    pop = self.toolbox.population(n=self.pop_size)
    hof = tools.HallOfFame(25)
    logbook = tools.Logbook()
    eval_smiles_fitness = partial(self.evaluate_fitness, best_isolate_representation=best_isolate_representation, model=model)
    pop_per_gen = [pop]

    # Evaluate the entire population
    if self.debug >= 1:
      print("Generating fitness for generation 0...")

    fitnesses = [eval_smiles_fitness(antibiotic.get_smiles()) for antibiotic in pop]

    # fitnesses = list(self.map_func(eval_antibiotic_seq_fitness, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        ind.generation = 0

    record = self.stats.compile(pop) if self.stats else {}
    logbook.record(gen=0, **record)
    best_every_iter = []

    for g in range(self.num_gen):
      # Select the next generation individuals (with replacement)
      if self.debug >= 1:
        print("Selecting...")

      offspring = self.toolbox.select(pop, len(pop))
      
      # Clone the selected individuals (since selection only took references rather than values)
      if self.debug >= 1:
        print("Cloning...")

      offspring = list(map(self.toolbox.clone, offspring))

      # Apply crossover on the offspring
      if self.debug >= 1:
        print("Crossover...")

      for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random() < self.cxpb:
          self.toolbox.mate(child1, child2)
          del child1.fitness.values
          del child2.fitness.values

      # Apply mutation on the offspring
      if self.debug >= 1:
        print("Mutating...")

      for mutant in offspring:
        self.toolbox.mutate(mutant)
        if mutant.get_if_changed():
          del mutant.fitness.values

      # Evaluate the individuals with an invalid fitness
      invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
      if self.debug >= 1:
        print("Calculating new fitnesses...")

      fitnesses = [eval_smiles_fitness(antibiotic.get_smiles()) for antibiotic in invalid_ind]

      if self.debug >= 2:
        self._print_fitnesses(g, fitnesses)
      elif self.debug == 1 and g % 10 == 0:
        self._print_fitnesses(g, fitnesses)

      for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        ind.generation = g + 1

      # The population is entirely replaced by the offspring
      if self.debug >= 1:
        print("Saving generation...")

      pop[:] = offspring
      pop_per_gen.append(pop)
      hof.update(pop)
      record = self.stats.compile(pop) if self.stats else {}
      logbook.record(gen=g + 1, **record)
      best_every_iter.append(hof[0])

    if self.debug >= 0:
      print(f"\tBest individual seen fitness value:\t\t{hof[0].fitness.values}")
      print(f"\tBest individual seen generation appeared in:\t{hof[0].generation}")

    gen, min_results, max_results, avg, std = logbook.select("gen", "min", "max", "avg", "std")

    return best_every_iter, {"iterations": gen, "min": min_results, "max": max_results, "avg": avg,
                                              "std": std}, pop_per_gen

  def __getstate__(self):
      self_dict = self.__dict__.copy()
      # del self_dict['map_func']
      return self_dict

  def __setstate__(self, state):
      self.__dict__.update(state)

  
def load_alg(smiles, **options) -> AntGenAlg:
  return AntGenAlg(smiles, **options)
