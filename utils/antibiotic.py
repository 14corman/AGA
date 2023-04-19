# -*- coding: utf-8 -*-
"""
Created on 21Oct2022 3:39 PM

@author: Cory Kromer-Edwards

Isolate Object to be used in Bacterial GA
"""

from random import choices
from . import ModSMI
from rdkit import Chem

class Antibiotic(object):
  def __init__(self, smiles, history=None) -> None:
    self.smiles = smiles
    self.is_changed = False
    if history is None:
      self.history = []
    else:
      self.history = history


  def get_smiles(self):
    return self.smiles
    

  def get_history(self):
    return self.history


  def get_if_changed(self):
    return self.is_changed


  def _replace_atom(self):
    try:
      new_smi, mol = ModSMI.replace_atom(self.smiles)
      if mol:
        self.history.append(self.smiles)
        self.smiles = new_smi
        self.is_changed = True
    except (PermissionError, Chem.rdchem.KekulizeException):
      pass
          

  def _add_atom(self):
    try:
      new_smi, mol = ModSMI.add_atom(self.smiles)
      if mol:
        self.history.append(self.smiles)
        self.smiles = new_smi
        self.is_changed = True
    except(PermissionError, Chem.rdchem.KekulizeException):
      pass
          

  def _delete_atom(self):
    try:
      new_smi, mol = ModSMI.delete_atom(self.smiles)
      if mol:
        self.history.append(self.smiles)
        self.smiles = new_smi
        self.is_changed = True
    except PermissionError:
      pass


  def mutate(self, add_prob=33, replace_prob=33, delete_prob=33):
    self.is_changed = False
    perform = choices([0, 1, 2], weights=[add_prob, replace_prob, delete_prob], k=1)[0]
    if perform == 0:
      self._add_atom()
    elif perform == 1:
      self._replace_atom()
    elif perform == 2:
      self._delete_atom()
    

