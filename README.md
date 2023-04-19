# Antibiotic Genetic Algorithm (AGA)
This is the base code for AGA in the BAAGA paper.

# Install
Install requires `seaborn`, `matplotlib`, `deap`, `rdkit`, and `numpy`.

# Running
This is not meant to run in the console. It is meant as a baseline for using Genetic Algorithms with Antibiotics to generate new compounds based on a fitness function. It is a barebones implementation with comments provided to give a jump start in using Genetic Algorithms for generating compounds.

# Citation
If you use any ideas or code from this repository, please cite the following paper where it was published:
```
@INPROCEEDINGS{Krom2306:Finding,
AUTHOR="Cory Kromer-Edwards and Suely Oliveira",
TITLE="Finding, and Countering, Future Resistance Using Bacterial Antibiotic
Adversarial Genetic Algorithm {(BAAGA)}",
BOOKTITLE="2023 International Joint Conference on Neural Networks (IJCNN) (IJCNN 2023)",
ADDRESS="Queensland, Australia",
DAYS="17",
MONTH=6,
YEAR=2023,
ABSTRACT="Since the beginning of antimicrobial therapy, antimicrobial resistance from
bacteria has been a threat. Now, multiple companies track antibiotic
resistance annually. Monitoring helps watch for new resistant mechanisms to
restrict spread. This monitoring also determines if a new antimicrobial
agent must be created. Proper prescription of an antimicrobial agent also
plays a key role in slowing resistance. Proper prescription to a patient
requires the determination of a Minimum Inhibitory Concentration (MIC).
These MICs are slow to determine, which raises morbidity and mortality
rates in hospital settings. Physicians cannot predict the creation of new
resistance mechanisms based on the prescription of one agent over another.
It is also impossible for researchers to know what agents to create in the
future. Because of this, there is a race against the clock when new
resistant bacteria emerge, to create appropriate antimicrobial agents to
counter the new mechanisms of resistance.  In this study, we propose a new
simulation algorithm named Bacterial Antibiotic Adversarial Genetic
Algorithm (BAAGA) that simulates the generation of new mechanisms of
resistance and new antimicrobial agents in a stochastic manner. BAAGA shows
the ease at which bacteria can gain resistance mechanisms and how slow the
process is to find new antimicrobial agents. We also demonstrate how BAAGA
can produce possible new antimicrobial agents to counter mechanisms of
resistance that currently do not exist."
}


```
