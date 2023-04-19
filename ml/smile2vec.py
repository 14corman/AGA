"""
This file contains all functions necessary to convert a compound's
smile string into an embeded vector representation.

Base for code was taken from the following files on Github:
- https://github.com/Abdulk084/Smiles2vec/blob/master/smiles2vec.ipynb

@author Cory Kromer-Edwards

Created: 9Aug2022 8:23 AM
"""

# import re
import numpy as np
# from sklearn.preprocessing import StandardScaler
import pandas as pd
import random
import rdkit.Chem as rkc

CHAR_TO_INT_CSV = "./ml/metadata/smile_char_to_int.csv"
SMILE_VARS = "./ml/metadata/smile_vars.csv"

class Smile2Vec(object):
  def __init__(self):
    vars_dict = pd.read_csv(SMILE_VARS, index_col="var").to_dict('index')
    char_to_int_dict = pd.read_csv(CHAR_TO_INT_CSV, index_col="char").to_dict('index')

    self.char_to_int = {k: v.get('int') for k, v in char_to_int_dict.items()}
    self.int_to_char = {v.get('int'): k for k, v in char_to_int_dict.items()}
    self.embed = int(vars_dict.get("embed").get("value"))
    self.charset = eval(vars_dict.get("charset").get("value"))

  def prepare_for_training(self, input_file, num_random_smiles_generations=10):
    input_df = pd.read_csv(input_file, index_col="antibiotics")

    smiles = [ list(s)[0] for s in input_df[["smiles"]].values]
    self.charset = set("".join(list(smiles)) + "!E")
    self.char_to_int = dict((c,i) for i,c in enumerate(self.charset))
    self.int_to_char = dict((i,c) for i,c in enumerate(self.charset))
    self.embed = max([len(smile) for smile in smiles]) + 5
    print (str(self.charset))
    print(len(self.charset), self.embed)

    pd.DataFrame([["embed", self.embed], ["charset", self.charset]], columns=["var", "value"]).to_csv(SMILE_VARS, index=False)
    pd.DataFrame(self.char_to_int.items(), columns=["char", "int"]).to_csv(CHAR_TO_INT_CSV, index=False)

    input_dict = input_df[["smiles"]].to_dict('index')

    # Vectorize smiles
    smiles_dict = dict()
    for drug in input_dict.keys():
      # Vectorize canonical smiles
      canonical_smiles = input_dict.get(drug).get("smiles")
      molecule = self.to_mol(canonical_smiles)

      # Generate random SMILES
      smiles_set = set([canonical_smiles])
      for _ in range(num_random_smiles_generations):
        rand_smiles = self.randomize_smiles(molecule)
        smiles_set.add(rand_smiles)

      one_hot_smiles_list = []
      for smiles in smiles_set:
        one_hot_smiles, _smiles_exceeds_embeding = self.prepare_smile(smiles)
        one_hot_smiles_list.append(one_hot_smiles)

      smiles_dict[drug] = one_hot_smiles_list
    
    return smiles_dict

  def prepare_smile(self, smile):
    one_hot =  np.zeros((self.embed , len(self.charset)), dtype=np.int8)
    
    #encode the startchar
    one_hot[0, self.char_to_int["!"]] = 1

    smiles_exceeds_embeding = False

    if len(smile) <= self.embed - 2:
      #encode the rest of the chars
      for j,c in enumerate(smile):
          one_hot[j+1 , self.char_to_int[c]] = 1

      #Encode endchar
      one_hot[len(smile)+1:, self.char_to_int["E"]] = 1
    else:
      #encode the rest of the chars
      for i in range(self.embed - 2):
          one_hot[i+1 , self.char_to_int[smile[i]]] = 1

      #Encode endchar
      one_hot[self.embed:, self.char_to_int["E"]] = 1
      smiles_exceeds_embeding = True

    return one_hot[0:-1, :], smiles_exceeds_embeding

  def to_mol(self, smi):
    """
    Creates a Mol object from a SMILES string.
    https://github.com/undeadpixel/reinvent-randomized
    :param smi: SMILES string.
    :return: A Mol object or None if it's not valid.
    """
    if smi:
        return rkc.MolFromSmiles(smi)

  def to_smiles(self, mol):
    """
    Converts a Mol object into a canonical SMILES string.
    https://github.com/undeadpixel/reinvent-randomized
    :param mol: Mol object.
    :return: A SMILES string.
    """
    return rkc.MolToSmiles(mol, isomericSmiles=False)

  def randomize_smiles(self, mol, random_type="restricted"):
    """
    Returns a random SMILES given a SMILES of a molecule.
    https://github.com/undeadpixel/reinvent-randomized
    :param mol: A Mol object
    :param random_type: The type (unrestricted, restricted) of randomization performed.
    :return : A random SMILES string of the same molecule or None if the molecule is invalid.
    """
    if not mol:
        return None

    if random_type == "unrestricted":
        return rkc.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=False)

    if random_type == "restricted":
        new_atom_order = list(range(mol.GetNumAtoms()))
        random.shuffle(new_atom_order)
        random_mol = rkc.RenumberAtoms(mol, newOrder=new_atom_order)
        return rkc.MolToSmiles(random_mol, canonical=False, isomericSmiles=False)

    raise ValueError("Type '{}' is not valid".format(random_type))
