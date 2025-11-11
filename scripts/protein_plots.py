"""
Protein Structure Prediction Visualization Module

This module provides comprehensive visualization functions for protein secondary structure
prediction models. It includes tools for visualizing model predictions, performance
metrics, and analysis of model behavior across different sequence lengths.

Functions:
    plot_model_predictions_on_dataset: Visualize model predictions on sample sequences
    plot_enhanced_confusion_matrix: Create detailed confusion matrix with per-class accuracy
    plot_length_performance: Analyze performance vs. sequence length
    plot_class_performance: Compare per-class metrics
    evaluate_with_visualizations: Generate all visualizations for a trained model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
import logging
from protein_struct_preprocess import preprocess_proteins
from protein_struct_model_prep import proteinDataset, pad_mask
from protein_struct_model import lstm_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def plot_model_predictions_on_dataset(test_data, model, prime2idx, lab3, num_samples=5, device="mps"):
    """
    Visualize model predictions on actual dataset samples.
    
    This function creates a comprehensive visualization showing true vs. predicted secondary
    structures for sample protein sequences, along with accuracy indicators.
    
    Args:
        test_data (pd.DataFrame): Test dataset containing protein sequences and labels
        model (nn.Module): Trained model for prediction
        prime2idx (dict): Vocabulary mapping for amino acids
        lab3 (dict): Label mapping for 3-class structure prediction
        num_samples (int, optional): Number of samples to visualize. Defaults to 5.
        device (str, optional): Device to use for computation. Defaults to "mps".
        
    Returns:
        None: Saves visualization to 'model_predictions_visualization.png'
    """
    # Create dataset and loader
    test_dataset = proteinDataset(data=test_data, prime2idx=prime2idx,
                              lab3=lab3, lab8=None, label_mode="q3", max_len=512)
    collate = lambda batch: pad_mask(batch, x_pad=prime2idx, y_pad=lab3)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)
    
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    with torch.no_grad():
        for i, (x, y, mask) in enumerate(test_loader):
            if i >= num_samples:
                break
                
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            # Get model prediction
            logits = model(x)
            pred = logits.argmax(-1)
            
            # Convert to labels
            true_seq = [lab3[idx.item()] for idx in y[0][mask[0]]]
            pred_seq = [lab3[idx.item()] for idx in pred[0][mask[0]]]
            
            # Get original sequence
            original_seq = test_data.iloc[i]['seq'][:len(true_seq)]
            
            # Plot 1: Sequence with true labels
            axes[i, 0].bar(range(len(true_seq)),
                              [2 if s == 'H' else 1 if s == 'E' else 0 for s in true_seq],
                              color=['red' if s == 'H' else 'yellow' if s == 'E' else 'grey' for s in true_seq])
            axes[i, 0].set_title(f'True Structure (Sample {i+1})')
            axes[i, 0].set_ylim(-0.1, 2.1)
            axes[i, 0].set_yticks([0, 1, 2], ['C', 'E', 'H'])
            
            # Plot 2: Sequence with predicted labels
            axes[i, 1].bar(range(len(pred_seq)),
                              [2 if s == 'H' else 1 if s == 'E' else 0 for s in pred_seq],
                              color=['red' if s == 'H' else 'yellow' if s == 'E' else 'grey' for s in pred_seq])
            axes[i, 1].set_title(f'Predicted Structure (Sample {i+1})')
            axes[i, 1].set_ylim(-0.1, 2.1)
            axes[i, 1].set_yticks([0, 1, 2], ['C', 'E', 'H'])
            
            # Plot 3: Accuracy visualization
            correct = [1 if t == p else 0 for t, p in zip(true_seq, pred_seq)]
            accuracy = sum(correct) / len(correct) if correct else 0
            
            colors = ['green' if c else 'red' for c in correct]
            axes[i, 2].bar(range(len(correct)), [1]*len(correct), color=colors)
            axes[i, 2].set_title(f'Accuracy: {accuracy:.2f}')
            axes[i, 2].set_ylim(-0.1, 1.1)
            axes[i, 2].set_yticks([])
            
            # Add amino acid labels
            for ax in axes[i]:
                ax.set_xticks(range(len(original_seq)))
                ax.set_xticklabels(list(original_seq), rotation=45)
                ax.set_xlabel('Amino Acid Position')
    
    plt.tight_layout()
    plt.savefig('model_predictions_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_enhanced_confusion_matrix(y_true, y_pred, lab3, save_path=None):
    """
    Create a more informative confusion matrix with per-class accuracy.
    
    This function generates a detailed confusion matrix visualization along with a bar chart
    showing per-class accuracy for better understanding of model performance.
    
    Args:
        y_true (list): True labels for all samples
        y_pred (list): Predicted labels for all samples
        lab3 (dict): Label mapping for structure prediction
        save_path (str, optional): Path to save the visualization. Defaults to None.
        
    Returns:
        None: Displays and optionally saves the visualization
    """
    # Get class names
    classes = [k for k in lab3.keys() if k != '<PAD>']
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Standard confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Per-class accuracy
    ax2.bar(classes, per_class_acc, color=['grey', 'yellow', 'red'])
    ax2.set_title('Per-Class Accuracy')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim(0, 1)
    
    # Add accuracy values on bars
    for i, acc in enumerate(per_class_acc):
        ax2.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_length_performance(test_data, model, prime2idx, lab3, device="mps"):
    """
    Analyze how model performance varies with sequence length.
    
    This function evaluates model performance across different sequence lengths to identify
    patterns or biases in model behavior with respect to input size.
    
    Args:
        test_data (pd.DataFrame): Test dataset containing protein sequences and labels
        model (nn.Module): Trained model for evaluation
        prime2idx (dict): Vocabulary mapping for amino acids
        lab3 (dict): Label mapping for structure prediction
        device (str, optional): Device to use for computation. Defaults to "mps".
        
    Returns:
        None: Saves visualization to 'length_performance.png'
    """
    # Create dataset and get predictions
    test_dataset = proteinDataset(data=test_data, prime2idx=prime2idx,
                              lab3=lab3, lab8=None, label_mode="q3", max_len=512)
    collate = lambda batch: pad_mask(batch, x_pad=prime2idx, y_pad=lab3)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)
    
    lengths = []
    accuracies = []
    
    model.eval()
    with torch.no_grad():
        for i, (x, y, mask) in enumerate(test_loader):
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            
            # Get prediction
            logits = model(x)
            pred = logits.argmax(-1)
            
            # Calculate accuracy for this sequence
            true_seq = y[0][mask[0]]
            pred_seq = pred[0][mask[0]]
            
            correct = (true_seq == pred_seq).float().mean().item()
            seq_len = mask[0].sum().item()
            
            lengths.append(seq_len)
            accuracies.append(correct)
    
    # Create bins for sequence lengths
    bins = np.linspace(min(lengths), max(lengths), 10)
    bin_indices = np.digitize(lengths, bins) - 1
    
    # Calculate average accuracy per bin
    bin_accuracies = []
    bin_centers = []
    for i in range(len(bins)):
        mask_bin = bin_indices == i
        if mask_bin.sum() > 0:
            bin_accuracies.append(np.mean(np.array(accuracies)[mask_bin]))
            bin_centers.append((bins[i] + (bins[i+1] if i+1 < len(bins) else bins[i]))/2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Sequence Length')
    plt.ylabel('Accuracy')
    plt.title('Model Performance vs. Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.savefig('length_performance.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_class_performance(metrics_dict, classes, save_path=None):
    """
    Plot per-class performance metrics for comparison.
    
    This function creates a bar chart comparing different metrics (precision, recall,
    F1-score) across different secondary structure classes.
    
    Args:
        metrics_dict (dict): Dictionary with metric names as keys and lists of values as values
        classes (list): List of class names
        save_path (str, optional): Path to save the visualization. Defaults to None.
        
    Returns:
        None: Displays and optionally saves the visualization
    """
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (metric, values) in enumerate(metrics_dict.items()):
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_xlabel('Secondary Structure Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_with_visualizations(model_path, test_data_path, prime2idx, lab3, device="mps"):
    """
    Load trained model and generate comprehensive visualizations.
    
    This function loads a trained model and generates multiple visualizations
    including confusion matrix, sample predictions, length performance analysis,
    and per-class metrics comparison.
    
    Args:
        model_path (str): Path to the saved model state
        test_data_path (str): Path to the test dataset
        prime2idx (dict): Vocabulary mapping for amino acids
        lab3 (dict): Label mapping for structure prediction
        device (str, optional): Device to use for computation. Defaults to "mps".
        
    Returns:
        None: Generates and saves multiple visualization files
    """
    # Load the trained model
    device = torch.device(device)
    
    # Create model with same architecture as during training
    model = lstm_model(vocab_size=prime2idx,
                      num_tags=len(lab3),
                      pad_id=prime2idx["<PAD>"],
                      hidden=20,
                      embed_dim=20,
                      bidir=True)
    
    # Load the saved state
    best_state = torch.load(model_path, map_location=device)
    model.load_state_dict(best_state)
    model.to(device)
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    
    logger.info("Generating model predictions...")
    
    # Get predictions for visualization
    test_dataset = proteinDataset(data=test_data, prime2idx=prime2idx,
                              lab3=lab3, lab8=None, label_mode="q3", max_len=512)
    collate = lambda batch: pad_mask(batch, x_pad=prime2idx, y_pad=lab3)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate)
    
    model.eval()
    all_p, all_y = [], []
    
    with torch.no_grad():
        for step, (x, y, mask) in enumerate(test_loader, 1):
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            logits = model(x)
            prob = logits.argmax(-1)
            
            # Keep only real tags, none of the PADDING
            for yi, pi, mi in zip(y, prob, mask):
                yi = yi[mi].tolist()
                pi = pi[mi].tolist()
                all_y.append(yi)
                all_p.append(pi)
    
    # Map ids -> tags
    idx2tag = {v:k for k,v in lab3.items()}
    true_tags = [idx2tag[tag] for sequence in all_y for tag in sequence]
    predict_tags = [idx2tag[tag] for sequence in all_p for tag in sequence]
    
    # Generate visualizations
    logger.info("Creating confusion matrix...")
    plot_enhanced_confusion_matrix(true_tags, predict_tags, lab3,
                                  save_path='enhanced_confusion_matrix.png')
    
    logger.info("Creating sample predictions visualization...")
    plot_model_predictions_on_dataset(test_data, model, prime2idx, lab3,
                                   num_samples=5, device=device)
    
    logger.info("Analyzing length performance...")
    plot_length_performance(test_data, model, prime2idx, lab3, device=device)
    
    # Create performance metrics
    from sklearn.metrics import classification_report
    report = classification_report(true_tags, predict_tags, digits=3, zero_division=0, output_dict=True)
    
    # Extract metrics for plotting
    metrics_dict = {
        'Precision': [report['C']['precision'], report['E']['precision'], report['H']['precision']],
        'Recall': [report['C']['recall'], report['E']['recall'], report['H']['recall']],
        'F1-Score': [report['C']['f1-score'], report['E']['f1-score'], report['H']['f1-score']]
    }
    
    logger.info("Creating performance comparison...")
    plot_class_performance(metrics_dict, ['C', 'E', 'H'], save_path='class_performance.png')
    
    logger.info("All visualizations completed successfully!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python protein_plots.py <model_path> <test_data_path>")
        print("Example: python protein_plots.py best_model_state_for_label3.pth /path/to/test.csv")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_data_path = sys.argv[2]
    
    # Load the preprocessing data to get vocabularies
    train_df = "/Users/mubarak/Projects/BioML/protein_struct_proj/dataset/training_secondary_structure_train.csv"
    _, prime2idx, lab3, _ = preprocess_proteins(train_df)
    
    # Run evaluation with visualizations
    evaluate_with_visualizations(model_path, test_data_path, prime2idx, lab3)
