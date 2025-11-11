"""
Protein Data Preprocessing Module

This module handles preprocessing of protein sequence data for secondary structure prediction.
It converts amino acid sequences and structure labels to integer representations and creates
vocabularies for model training.

Functions:
    preprocess_proteins: Load protein data and create vocabularies for sequences and labels
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
import logging, os, time
import torch
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def preprocess_proteins(df):
    """
    Load and preprocess protein sequence data for secondary structure prediction.
    
    This function reads a CSV file containing protein sequences and their secondary
    structure labels, then creates vocabularies to convert amino acids and structure
    labels to integer representations suitable for neural network training.
    
    Args:
        df (str): Path to the CSV file containing protein data. The file should
                  contain columns named 'seq' (amino acid sequences), 'sst3'
                  (3-class secondary structure labels), and 'sst8'
                  (8-class secondary structure labels).
    
    Returns:
        tuple: A tuple containing:
            - data (pd.DataFrame): The loaded protein data
            - prim2idx (dict): Mapping from amino acids to integer indices
            - lab3 (dict): Mapping from 3-class structure labels to indices
            - lab8 (dict): Mapping from 8-class structure labels to indices
    
    Example:
        >>> data, prime2idx, lab3, lab8 = preprocess_proteins("training_data.csv")
        >>> print(f"Vocabulary size: {len(prime2idx)}")
        >>> print(f"3-class labels: {lab3}")
    """
    # Load the protein data from CSV file
    data = pd.read_csv(df, sep=",")

    # Create vocabulary for amino acid sequences
    # The first column 'seq' contains the input sequences
    # We convert each amino acid to a unique integer ID (similar to word2idx)
    prim2idx = {"<PAD>":0 , "<UNK>":1}

    # Iterate through all sequences and amino acids to build the vocabulary
    for seq in data['seq']:  # Access each sequence in the column
        for aa in seq:  # Access each amino acid in that sequence
            
            aa = aa.upper()  # Convert to uppercase for consistency
            
            # Add new amino acids to the vocabulary
            if aa not in prim2idx:
                prim2idx[aa] = len(prim2idx)

    # Create a preview of the vocabulary for debugging
    preview = {k: prim2idx[k] for k in list(prim2idx)[:12]}

    # Create label mappings for secondary structure predictions
    # Columns 2/3 (sst3, sst8) contain the target labels we want to predict
    # Extract unique characters from the structure labels
    label3_chars = sorted({ch for s in data["sst3"].astype(str) for ch in s })  # Results in ['C', 'E', 'H']
    label8_chars = sorted({ch for s in data["sst8"].astype(str) for ch in s})

    # Create label-to-index mappings with padding token at index 0
    # Using dict unpacking (**) to combine padding token with label mappings
    lab3 = {"<PAD>":0, **{c: i+1  for i, c in enumerate(label3_chars)}}  # 3-class structure labels
    lab8 = {"<PAD>":0, **{c: i+1 for i, c in enumerate(label8_chars)}}  # 8-class structure labels
   
    return data, prim2idx, lab3, lab8

#test the mapping
#df = r"/Users/mubarak/Projects/BioML/protein_struct_proj/dataset/training_secondary_structure_train.csv"
#preprocess_proteins(df)

