"""
Protein Dataset and Padding Utilities

This module provides the Dataset class and padding utilities for protein sequence data.
It handles the conversion of preprocessed protein data into PyTorch tensors suitable
for model training, including padding and masking for variable-length sequences.

Classes:
    proteinDataset: PyTorch Dataset for protein sequence and structure data

Functions:
    pad_mask: Pad sequences and create attention masks for batching
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging, os, time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

class proteinDataset(Dataset):
    """
    PyTorch Dataset for protein sequence and secondary structure data.
    
    This dataset handles the conversion of protein sequences and their corresponding
    secondary structure labels to integer tensors suitable for neural network training.
    It supports both 3-class (Q3) and 8-class (Q8) secondary structure prediction.
    
    Attributes:
        data (pd.DataFrame): Protein sequence and structure data
        prime2idx (dict): Mapping from amino acids to integer indices
        lab3 (dict): Mapping from 3-class structure labels to indices
        lab8 (dict): Mapping from 8-class structure labels to indices
        label_mode (str): Either "q3" or "q8" to specify prediction task
        max_len (int, optional): Maximum sequence length for truncation
    """
    
    def __init__(self, data, prime2idx, lab3, lab8, label_mode="q3", max_len=None):
        """
        Initialize the protein dataset.
        
        Args:
            data (pd.DataFrame): DataFrame containing protein sequences and structure labels
            prime2idx (dict): Mapping from amino acids to integer indices
            lab3 (dict): Mapping from 3-class structure labels to indices
            lab8 (dict): Mapping from 8-class structure labels to indices
            label_mode (str, optional): Prediction mode, either "q3" or "q8". Defaults to "q3".
            max_len (int, optional): Maximum sequence length for truncation. Defaults to None.
        """
        self.data = data.reset_index(drop=True)
        self.prime2idx = prime2idx
        self.lab3 = lab3
        self.lab8 = lab8

        assert label_mode in ("q3", "q8")
        self.label_mode = label_mode
        self.max_len = max_len

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of protein sequences in the dataset
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Get a single sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Integer-encoded amino acid sequence
                - torch.Tensor: Integer-encoded secondary structure labels
        """
        # Get the row at the specified index
        row = self.data.iloc[index]
        seq = str(row["seq"]).upper()
        sst3 = str(row["sst3"])
        sst8 = str(row["sst8"])

        # Ensure sequence and labels have the same length
        assert len(seq) == len(sst3) == len(sst8)

        # Optional truncation to keep samples bounded (useful for memory/batching)
        if self.max_len is not None:
            seq = seq[:self.max_len]
            sst3 = sst3[:self.max_len]
            sst8 = sst8[:self.max_len]
        
        # Map amino acid characters to integer IDs
        # Using .get(key, default) to handle unknown amino acids with <UNK> token
        # This makes the input data robust against unexpected characters
        seq_ids = [self.prime2idx.get(aa, self.prime2idx["<UNK>"]) for aa in seq]

        # Map structure labels to integer IDs based on the selected label mode
        if self.label_mode == "q3":
            y_ids = [self.lab3[ch] for ch in sst3]  # 3-class structure labels
            pad_y = self.lab3["<PAD>"]
        else:
            y_ids = [self.lab8[ch] for ch in sst8]  # 8-class structure labels
            pad_y = self.lab8["<PAD>"]

        return (torch.tensor(seq_ids, dtype=torch.long),
                torch.tensor(y_ids, dtype=torch.long))
         

def pad_mask(batch, x_pad, y_pad):
    """
    Pad sequences in a batch and create attention masks.
    
    This function takes a batch of variable-length sequences and pads them to the
    same length, while also creating a mask to distinguish real tokens from padding.
    
    Args:
        batch (list): List of tuples, where each tuple contains (sequence_tensor, labels_tensor)
        x_pad (dict): Vocabulary dictionary for sequences containing "<PAD>" key
        y_pad (dict): Vocabulary dictionary for labels containing "<PAD>" key
        
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Padded sequences [batch_size, max_seq_len]
            - torch.Tensor: Padded labels [batch_size, max_seq_len]
            - torch.Tensor: Attention mask [batch_size, max_seq_len] (1 for real tokens, 0 for padding)
    
    Example:
        >>> batch = [(seq_tensor1, label_tensor1), (seq_tensor2, label_tensor2)]
        >>> seqs, labels, mask = pad_mask(batch, prime2idx, lab3)
    """
    # Unzip the batch to separate sequences and labels
    seqs, labels = zip(*batch)

    # Pad sequences and labels to match the longest sequence in the batch
    seq_padded = pad_sequence(seqs, batch_first=True, padding_value=x_pad["<PAD>"])
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=y_pad["<PAD>"])

    # Create attention mask: 1 for real tokens, 0 for padding
    # This helps the model ignore padding during computation
    mask = (seq_padded != x_pad["<PAD>"])

    return seq_padded, labels_padded, mask

