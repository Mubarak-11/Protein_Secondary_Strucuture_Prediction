"""
Protein Secondary Structure Prediction Model

This module implements a bidirectional LSTM model for protein secondary structure prediction.
It includes model architecture, training functions, and evaluation utilities for predicting
protein secondary structures from amino acid sequences.

Classes:
    lstm_model: Bidirectional LSTM model for protein secondary structure prediction

Functions:
    protein_train: Train the protein structure prediction model
    evaluate: Evaluate model performance on test data
    main: Main function to run training and evaluation pipeline
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import logging, time, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import torch.optim.lr_scheduler
from sklearn.pipeline import Pipeline
from collections import Counter
import json
import torch.nn as nn
from protein_struct_preprocess import preprocess_proteins
from protein_struct_model_prep import proteinDataset, pad_mask
from sklearn.metrics import f1_score, precision_recall_curve, accuracy_score, roc_auc_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
logger.info(f"Using Apple {device}")


class lstm_model(nn.Module):
    """
    Bidirectional LSTM model for protein secondary structure prediction.
    
    This model processes amino acid sequences through an embedding layer, followed by
    a bidirectional LSTM, layer normalization, dropout, and a final linear layer
    to predict secondary structure labels for each amino acid position.
    
    Attributes:
        pad_id (int): Padding token index
        embed (nn.Embedding): Embedding layer for amino acid sequences
        lstm (nn.LSTM): Bidirectional LSTM layer
        norm (nn.LayerNorm): Layer normalization
        dropout (nn.Dropout): Dropout layer for regularization
        final_layer (nn.Linear): Final linear layer for classification
    """
    
    def __init__(self, vocab_size, num_tags, pad_id, hidden=64, embed_dim=64, bidir=False):
        """
        Initialize the LSTM model.
        
        Args:
            vocab_size (dict): Vocabulary dictionary for amino acids
            num_tags (int): Number of output classes (secondary structure types)
            pad_id (int): Padding token index
            hidden (int, optional): Hidden dimension of LSTM. Defaults to 64.
            embed_dim (int, optional): Embedding dimension. Defaults to 64.
            bidir (bool, optional): Whether to use bidirectional LSTM. Defaults to False.
        """
        super().__init__()
        
        self.pad_id = pad_id
        self.embed = nn.Embedding(num_embeddings=len(vocab_size), embedding_dim=embed_dim, padding_idx=pad_id)
        
        # Two-layer LSTM with bidirectional processing
        # Bidirectional LSTM processes the sequence forwards and backwards and concatenates the outputs
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden, num_layers=2, batch_first=True, bidirectional=bidir) 

        # Output dimension doubles if bidirectional is True
        out_dim = hidden * (2 if bidir else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=0.2)  # Optional, but can help with stability

        # Final linear layer maps to output classes
        self.final_layer = nn.Linear(out_dim, num_tags)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Output logits of shape [batch_size, seq_len, num_classes]
        """
        # Input: [B, T] where B = batch size, T = sequence length (tokens)
        e = self.embed(x)  # Embedding layer: [B, T, E] where E = embedding dimension
        h, _ = self.lstm(e)  # LSTM layer: [B, T, H * (1 | 2)] where H = hidden dimension
        h = self.norm(h)  # Layer normalization across dimensions
        h = self.dropout(h)
        logits = self.final_layer(h)  # Final layer: [B, T, C] where C = number of classes

        return logits


def protein_train(lab3, model, train_loader, val_loader, n_factors=30, n_epochs=20, batch_size=64, label_mode='q3', device="mps"):
    """
    Train the protein secondary structure prediction model.
    
    This function implements a comprehensive training loop with gradient accumulation,
    learning rate scheduling, early stopping, and model checkpointing.
    
    Args:
        lab3 (dict): Label vocabulary for 3-class structure prediction
        model (nn.Module): The model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        n_factors (int, optional): Number of factors (unused). Defaults to 30.
        n_epochs (int, optional): Maximum number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 64.
        label_mode (str, optional): Label mode ('q3' or 'q8'). Defaults to 'q3'.
        device (str, optional): Device to use for training. Defaults to "mps".
        
    Returns:
        tuple: A tuple containing:
            - model (nn.Module): The trained model
            - best_model_state (dict): Best model state dictionary
    """
    # Initialize optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Learning rate scheduler that reduces LR when validation loss plateaus
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5, 
        patience=2
    )

    # Loss function that ignores padding tokens
    pad_y = lab3["<PAD>"]
    criteron = nn.CrossEntropyLoss(ignore_index=pad_y, reduction="sum")
    model.to(device)

    # Gradient accumulation to simulate larger batch sizes
    accum_steps = 4
    log_every = 100

    # Early stopping and model checkpointing variables
    best_loss = float("inf")
    best_model_state = None
    no_improve_epochs = 0
    train_epoch_loss = []
    val_epoch_loss = []

    for epoch in range(n_epochs):
        model.train()

        # Keep running sums on device to avoid per-step .item() calls
        # This keeps everything on the GPU for faster training
        # Avoids GPU↔CPU synchronization which slows training and breaks async execution
        train_running_loss_times_n = torch.tensor(0.0, device=device)
        train_running_n = torch.tensor(0.0, device=device)

        val_running_loss_times_n = torch.tensor(0.0, device=device)
        val_running_n = torch.tensor(0.0, device=device)

        optimizer.zero_grad(set_to_none=False)

        # Training loop
        for step, (x, y, mask) in enumerate(train_loader, 1):  # x:[B,T], y:[B,T], mask:[B,T]
            x = x.to(device, non_blocking=True)  # Kick off memory transfer asynchronously
            y = y.to(device, non_blocking=True)
            mask = mask.to(device)

            logit = model(x)    # [B, T, num_tags]
            B, T, C = logit.shape  # C = number of classes
            
            # Cross entropy loss expects [N,C] vs [N], so we reshape
            # SUM over non-PAD tokens
            loss_sum = criteron(logit.reshape(B*T, C), y.reshape(B*T))

            # Count actual non-pad tokens for normalization
            num_non_pad = mask.sum()

            # Normalize for gradient accumulation
            loss_normalized = loss_sum / (num_non_pad * accum_steps)
            
            loss_normalized.backward()  # Backward pass - scale grads only

            # Accumulate statistics
            with torch.no_grad():
                train_running_loss_times_n += loss_sum.detach() 
                train_running_n += num_non_pad

            # Update weights after accumulating gradients
            if step % accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
                optimizer.step()  # Update the weights
                optimizer.zero_grad(set_to_none=False)  # More memory efficient
            
        # Validation phase
        model.eval()
        with torch.no_grad():
            for step, (x, y, mask) in enumerate(val_loader, 1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                mask = mask.to(device)

                logit = model(x)
                B, T, C = logit.shape

                loss_sum = criteron(logit.reshape(B*T, C), y.reshape(B*T))
                
                non_pad_mask = mask.sum()  # Number of non-padded tokens

                val_running_loss_times_n += loss_sum  # No accumulation needed for validation
                val_running_n += non_pad_mask
            
        
        # Synchronize for MPS devices
        if device == "mps":
            torch.mps.synchronize()
                
        # Calculate average losses
        avg_train_loss = (train_running_loss_times_n / train_running_n).item() 
        avg_val_loss = (val_running_loss_times_n / val_running_n).item()  # .item() extracts Python scalar from tensor

        train_epoch_loss.append(avg_train_loss)
        val_epoch_loss.append(avg_val_loss)

        # Update learning rate based on validation loss
        sched.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logging.info(f" Epoch {epoch+1} / {n_epochs} | Train loss: {avg_train_loss:.4f},  Validation Loss: {avg_val_loss:.4f}, LR: {current_lr}")
        
        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            no_improve_epochs = 0
            best_model_state = model.state_dict()  # Save the best model weights
        
        else:
            no_improve_epochs += 1
        
        if no_improve_epochs >= 5:
            logging.info(f"No improvement after 5 epochs! Stopping at {epoch+1}")
        
    # Load best model and save checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        if label_mode == 'q3':
            torch.save(best_model_state, "best_model_state_for_label3.pth")
            logger.info("Best model state saved to best_model_state_for_label3.pth")
        else:
            torch.save(best_model_state, "best_model_state_for_label8.pth")
            logger.info("Best model state saved to best_model_state_for_label8.pth")

    # Plot training and validation loss
    plt.plot(train_epoch_loss, label='Train Loss')
    plt.plot(val_epoch_loss, label='val_loss')
    plt.title("Training and Validation Loss")
    plt.legend(); plt.grid()
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.show()

    return model, best_model_state


def evaluate(lab3, test_loader, model, batch_size=256, device="mps"):
    """
    Evaluate the trained model on test data.
    
    This function evaluates the model performance on a test dataset and generates
    a classification report with precision, recall, and F1-score metrics.
    
    Args:
        lab3 (dict): Label vocabulary for structure prediction
        test_loader (DataLoader): Test data loader
        model (nn.Module): Trained model to evaluate
        batch_size (int, optional): Batch size for evaluation. Defaults to 256.
        device (str, optional): Device to use for evaluation. Defaults to "mps".
    """
    pad_y = lab3["<PAD>"]
    criteron = nn.CrossEntropyLoss(ignore_index=pad_y)
    all_p, all_y = [], []

    # Enter evaluation mode
    model.eval()
    with torch.no_grad():
        for step, (x, y, mask) in enumerate(test_loader, 1):
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            logits = model(x)  # [B, T, C]
            prob = logits.argmax(-1)  # [B, T]
            
            # Keep only real tags, none of the padding
            for yi, pi, mi in zip(y, prob, mask):
                yi = yi[mi].tolist()
                pi = pi[mi].tolist()

                all_y.append(yi)
                all_p.append(pi)
    
    # Map IDs back to tags (skip pads)
    idx2tag = {v: k for k, v in lab3.items()}  # Reverse mapping from indices to tags

    # Flatten the nested lists before mapping
    true_tags = [idx2tag[tag] for sequence in all_y for tag in sequence]
    predict_tags = [idx2tag[tag] for sequence in all_p for tag in sequence]

    # Generate and print classification report
    logger.info(f" \nClassification report:\n{classification_report(true_tags, predict_tags, digits=3, zero_division=0)}")


def main():
    """
    Main function to run the complete training and evaluation pipeline.
    
    This function loads the data, initializes the model, trains it, and evaluates
    its performance on test data. It supports both 3-class (Q3) and 8-class (Q8)
    secondary structure prediction.
    """
    # Dataset file paths
    train_df = r"/Users/mubarak/Projects/BioML/protein_struct_proj/dataset/training_secondary_structure_train.csv"
    val_df = r"/Users/mubarak/Projects/BioML/protein_struct_proj/dataset/validation_secondary_structure_valid.csv"
    test_df = r"/Users/mubarak/Projects/BioML/protein_struct_proj/dataset/test_secondary_structure_casp12.csv"

    # Preprocess training data to create vocabularies
    t_data, t_prime2idx, t_lab3, t_lab8 = preprocess_proteins(train_df)
    v_data = pd.read_csv(val_df)  # Validation set
    ts_data = pd.read_csv(test_df)  # Test set

    # Setup collate function for padding and masking
    collate = lambda batch: pad_mask(batch, x_pad=t_prime2idx, y_pad=t_lab3)
    
    # Create datasets and data loaders
    train_dataset = proteinDataset(data=t_data, prime2idx=t_prime2idx, lab3=t_lab3, lab8=t_lab8, label_mode="q8", max_len=512)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate, num_workers=0)
    
    val_dataset = proteinDataset(data=v_data, prime2idx=t_prime2idx, lab3=t_lab3, lab8=t_lab8, label_mode='q8', max_len=512)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate, num_workers=0)

    test_dataset = proteinDataset(data=ts_data, prime2idx=t_prime2idx, lab3=t_lab3, lab8=t_lab8, label_mode='q8', max_len=512)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate, num_workers=0)
    
    # Initialize model with specified architecture
    base_model = lstm_model(vocab_size=t_prime2idx, 
                            num_tags=len(t_lab8),  # Number of output classes
                            pad_id=t_prime2idx["<PAD>"],
                            hidden=20,  # Hidden dimension
                            embed_dim=20,  # Embedding dimension
                            bidir=True)  # Use bidirectional LSTM

    base_model = base_model.to(device)

    # Model training (commented out to use pre-trained model)
    # model_train, best_model_state = protein_train(t_lab8, base_model, train_loader, val_loader, 
    #                                              n_factors=30, n_epochs=20, batch_size=512, 
    #                                              label_mode='q8', device="mps")

    # Load pre-trained model
    # For Q3:
    # best_state = torch.load("best_model_state_for_label3.pth", map_location=device)
    
    # For Q8:
    best_state = torch.load("best_model_state_for_label8.pth", map_location=device)
    
    base_model.load_state_dict(best_state)
    base_model.to(device)

    # Evaluate model on test data
    evaluate(lab3=t_lab8, test_loader=test_loader, model=base_model, batch_size=512, device="mps")


if __name__=="__main__":
    main()
