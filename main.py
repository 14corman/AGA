# -*- coding: utf-8 -*-
"""
Created on 8Dec2022 8:46 PM

@author: Cory Kromer-Edwards

Main entrypoint to the simulator (Manually updated after GridSearch)
"""

# import bact_alg
import ab_alg
# from ml import ClassifierModel, pred_to_log_mic
import re
import gzip
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd

# Seaborn has five built-in themes to style its plots: darkgrid, whitegrid, dark, white, and ticks. Seaborn defaults to using the darkgrid theme
sns.set_style("ticks")

# In order of relative size they are: paper, notebook, talk, and poster. The notebook style is the default.
sns.set_context("paper")

FASTA_PARSE_REGEX = re.compile('\>.+?\n([\w\n]+)')
NUM_ROUNDS = 3
# ANTIBIOTIC_OPTIONS = [("Aztreonam", "CC1C(C(=O)N1S(=O)(=O)O)NC(=O)C(=NOC(C)(C)C(=O)O)C2=CSC(=N2)N"), ("Meropenem", "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)N(C)C)C(=O)O)C(C)O"), ("Ceftriaxone", "CN1C(=NC(=O)C(=O)N1)SCC2=C(N3C(C(C3=O)NC(=O)C(=NOC)C4=CSC(=N4)N)SC2)C(=O)O")]
ANTIBIOTIC_OPTIONS = [("Aztreonam", "CC1C(C(=O)N1S(=O)(=O)O)NC(=O)C(=NOC(C)(C)C(=O)O)C2=CSC(=N2)N"), ("Meropenem", "CC1C2C(C(=O)N2C(=C1SC3CC(NC3)C(=O)N(C)C)C(=O)O)C(C)O")]
OUTPUT_DIR = "output"


def run_antibiotic_ga(antibiotic_ga, isolate_representation, model):
  best_every_iter, meta_dict, pop_per_gen = antibiotic_ga.run(isolate_representation, model)
  return best_every_iter, meta_dict["avg"], pop_per_gen


def parse_fasta(file_path):
  """Convert a Fasta file (may be compressed) into an embeding matrix representation.

  Args:
      file_path (str): File path to Fasta file

  Returns:
      [str]: List of sequences from fasta
  """
  sequences = []

  if file_path.endswith(".gz"):
    with gzip.open(file_path, 'rt') as fasta_file:
      try:
        file_contents = fasta_file.read()
        sequences = re.findall(FASTA_PARSE_REGEX, file_contents)
      except:
        raise(f"Error while reading fasta sequence from gziped file: {file_path}")
  else:
    with open(file_path, 'r') as fasta_file:
      file_contents = fasta_file.read()
      sequences = re.findall(FASTA_PARSE_REGEX, file_contents)

  sequences = [seq.replace('\n', '').upper() for seq in sequences]
  return sequences


def get_line_plot_legend_handles():
  algorithm_header = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label="Algorithm")
  isolate_line = mlines.Line2D([], [], color='blue', label='Isolate')
  bacteria_line = mlines.Line2D([], [], color='orange', label='Bacteria')
  result_header = mpatches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0, label="Result Value")
  mic_line = mlines.Line2D([], [], color='black', label='MIC Index')
  confidence_line = mlines.Line2D([], [], color='black', dashes=(2, 2), label='Confidence')
  return [algorithm_header, isolate_line, bacteria_line, result_header, mic_line, confidence_line]


def run_simulation(
    round_num,            # The round number for the simulation's parameters
    antibiotic_name,      # The antibiotic name being tested
    model,                # The ML model used to classify
    best_smiles,          # Seed for antibiotic GA
    chosen_fasta_file,    # Seed for bacteria GA
    # antibiotic_pop_size,  # Pop size for antibiotic GA
    # bacteria_pop_size,    # Pop size for bacteria GA
    # num_gen,              # Number of rounds to run GA's
    # snp_perc, indel_perc, bacteria_indpb, bacteria_tournsize, bacteria_cxpb,                    # Extra params for bacteria GA
    # add_prob, replace_prob, delete_prob, prob_keep_ring, antibiotic_tournsize, antibiotic_cxpb  # Extra params for antibiotic GA
  ):

  fasta_path = "./data/"
  iteration_best_mics = []
  iteration_avg_log_mics = []
  iteration_best_log_mics = []
  iteration_num = 0
  segment_num = 0  

  # Make the starting isolate representation to kick of Antibiotic GA
  # isolate_contigs = parse_fasta(fasta_path + chosen_fasta_file)
  # _contigs, best_representation = model.build_representation_from_seqs(isolate_contigs)
  # starting_log_mic = pred_to_log_mic(model.predict_for_smiles(best_representation, best_smiles)[0])
  starting_log_mic = 5
  best_representation = None

  for sim_round in range(NUM_ROUNDS):
    # Run Antibiotic GA and process results
    # antibiotic_ga = ab_alg.load_alg(best_smiles, pop_size=antibiotic_pop_size, num_gen=num_gen, add_prob=add_prob, replace_prob=replace_prob, delete_prob=delete_prob, prob_keep_ring=prob_keep_ring, tournsize=antibiotic_tournsize, cxpb=antibiotic_cxpb)
    antibiotic_ga = ab_alg.load_alg(best_smiles)
    best_ant_every_iter, average_log_mic, pop_per_gen = run_antibiotic_ga(antibiotic_ga, best_representation, model)
    # best_one_hot_smiles used for bacteria GA setup
    best_one_hot_smiles = model.build_one_hot_from_smiles(best_ant_every_iter[-1].get_smiles())
    best_smiles = best_ant_every_iter[-1].get_smiles()
    for iteration, (best_ant, avg_log_mic) in enumerate(zip(best_ant_every_iter, average_log_mic)):
      iteration_avg_log_mics.append([iteration, segment_num, "Antibiotic", avg_log_mic])
      # iteration_best_log_mics.append([iteration, segment_num, "Antibiotic", pred_to_log_mic(best_ant.fitness.values[0])])

    for pop in pop_per_gen:
      for p in pop:
        iteration_best_mics.append([iteration_num, segment_num, "Antibiotic", p.fitness.values[0]])

      iteration_num = iteration_num + 1

    segment_num = segment_num + 1

    # Run Bacteria GA and process results
    # ...

  best_log_mics_df = pd.DataFrame(iteration_best_log_mics, columns=["Iteration", "Segment", "Algorithm", "Log MIC"])
  average_log_mics_df = pd.DataFrame(iteration_avg_log_mics, columns=["Iteration", "Segment", "Algorithm", "Log MIC"])
  best_mics_df = pd.DataFrame(iteration_best_mics, columns=["Iteration", "Segment", "Algorithm", "MIC Index"])

  # AVERAGE MICS
  # Initialize a grid of plots with an Axes for each walk
  grid = sns.FacetGrid(average_log_mics_df, col="Segment", hue="Algorithm", palette={"Isolate": "blue", "Antibiotic": "orange"},
                      col_wrap=2, height=3.5, aspect=2)

  # Draw a horizontal line to show the starting point
  grid.refline(y=starting_log_mic, linestyle=":")

  # Draw a line plot to show the trajectory of each random walk
  grid.map(plt.plot, "Iteration", "Log MIC", marker="o")

  # Adjust the tick positions and labels
  grid.set(xticks=np.arange(0, 150, 10), yticks=np.arange(-7, 12, 2),
          xlim=(-.5, 150.5), ylim=(-7.5, 11.5))

  # Adjust the arrangement of the plots
  grid.fig.tight_layout(w_pad=1)
  plt.savefig(f"{OUTPUT_DIR}/{round_num}_{antibiotic_name}_average_log_mic_plots.pdf", bbox_inches='tight')
  plt.clf()
  
  # BEST MICS
  # Initialize a grid of plots with an Axes for each walk
  grid = sns.FacetGrid(best_log_mics_df, col="Segment", hue="Algorithm", palette={"Isolate": "blue", "Antibiotic": "orange"},
                      col_wrap=2, height=3.5, aspect=2)

  # Draw a horizontal line to show the starting point
  grid.refline(y=starting_log_mic, linestyle=":")

  # Draw a line plot to show the trajectory of each random walk
  grid.map(plt.plot, "Iteration", "Log MIC", marker="o")

  # Adjust the tick positions and labels
  grid.set(xticks=np.arange(0, 150, 10), yticks=np.arange(-7, 12, 2),
          xlim=(-.5, 150.5), ylim=(-7.5, 11.5))

  # Adjust the arrangement of the plots
  grid.fig.tight_layout(w_pad=1)
  plt.savefig(f"{OUTPUT_DIR}/{round_num}_{antibiotic_name}_best_log_mic_plots.pdf", bbox_inches='tight')
  plt.clf()
  
  best_log_mics_df.to_csv(f"{OUTPUT_DIR}/{round_num}_{antibiotic_name}_best_log_mics.csv", index=False)
  average_log_mics_df.to_csv(f"{OUTPUT_DIR}/{round_num}_{antibiotic_name}_average_log_mics.csv", index=False)
  best_mics_df.to_csv(f"{OUTPUT_DIR}/{round_num}_{antibiotic_name}_best_mics.csv", index=False)

def main():
  chosen_fasta_file = "Sentry-2016-949700_contigs.fasta"                # Seed isolate
  # model = ClassifierModel()
  model = None

  # Tuned hyperparameters
  # ...

  # Run each antibiotic simulation 3 times to get a sense of results.
  for round_num in range(3):
    for antibiotic_name, antibiotic_smiles in ANTIBIOTIC_OPTIONS:
      run_simulation(
        round_num,
        antibiotic_name,
        model,                
        antibiotic_smiles,          
        chosen_fasta_file,    
        # antibiotic_pop_size,  
        # bacteria_pop_size,    
        # num_gen,              
        # snp_perc, indel_perc, bacteria_indpb, bacteria_tournsize, bacteria_cxpb,
        # add_prob, replace_prob, delete_prob, prob_keep_ring, antibiotic_tournsize, antibiotic_cxpb
      )


if __name__ == '__main__':
  main()