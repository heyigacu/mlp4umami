import os
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_path, max_epochs=500, patience=7, delta=0, verbose=False, save_model=False):
        """
        Args:
            save_path (string): Model save path.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.save_path = save_path
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.max_epochs = max_epochs

        self.early_stop = False
        
        self.counter = 0
        self.best_score = None
        self.best_epoch = 1
        self.best_metrics = None
        self.save_model = save_model

    def __call__(self, epoch, model, val_loss, val_metric):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save_path != None and self.save_model:
                self.save_checkpoint(model)
            
        elif score < self.best_score + self.delta:
            self.counter += 1

        else:
            self.best_score = score
            self.best_epoch = epoch
            self.best_metric = val_metric
            if self.save_path != None and self.save_model:
                self.save_checkpoint(model)
            self.counter = 0
        
        if self.counter >= self.patience or epoch >= self.max_epochs:
            self.early_stop = True

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.save_path)	

