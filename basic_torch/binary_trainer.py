import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
import numpy as np
import os
from tqdm import tqdm
import copy

class BinaryTrainer:
    def __init__(self, model, x_train, y_train, x_val, y_val, 
                 batch_size=128, epochs=100, learning_rate=0.001,
                 optimizer=None, criterion=None, scheduler=None,
                 early_stopping_patience=5, device=None,
                 model_save_path='best_model.pth'):
        """
        pytorch trainer object designed for binary classification to mimic transformers trainer
        setups the training steps
        executes train
        eval against validation dataset to determine early stopping
        saves best model at each validation (no clawback needed if last model is worse than model[-5])
        
        Inputs:
            model: A binary PyTorch model (don't use signoid activation function in the last layer!)
            x_train: Training features df or np.array
            y_train: Training targets binary 0 1
            x_val: Validation features df or array 
            y_val: Validation targets binary 0 1
            batch_size: 64+
            epochs: ~100. Early stopping will end it early, but it does affect scheduler
            learning_rate: 0.001 is good. Adam optimizer will adjust
            optimizer: Adam or equivalent. Adjust learning_rate and scheduler if not adam.
            criterion: default is BCEWithLogitsLoss. Cross entrophy works too but need to add signoid activation to model.
            scheduler: default is CosineAnnealingLR with T_Max = max epochs for consistent scaling
            early_stopping_patience: Number of epochs to wait before early stopping
            device: 'cuda', 'cpu. Auto selects if left to none.
            model_save_path: Path to save the best model
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Convert data to PyTorch tensors
        if not isinstance(x_train, torch.Tensor):
            x_train = torch.tensor(x_train.values if hasattr(x_train, 'values') else x_train, dtype=torch.float32)
        if not isinstance(y_train, torch.Tensor):
            y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        if not isinstance(x_val, torch.Tensor):
            x_val = torch.tensor(x_val.values if hasattr(x_val, 'values') else x_val, dtype=torch.float32)
        if not isinstance(y_val, torch.Tensor):
            y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            
        # Create data loaders
        self.train_dataloader = DataLoader(
            TensorDataset(x_train, y_train),
            batch_size=batch_size,
            shuffle=True
        )
        
        self.val_dataloader = DataLoader(
            TensorDataset(x_val, y_val),
            batch_size=batch_size,
            shuffle=False
        )
        
        # device type
        self.model = model.to(self.device)
        
        # Set training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.model_save_path = model_save_path
        
        # Set optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
            
        # Set loss function (defaults to BCE)
        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion
            
        # Set scheduler if none, create with default (make sure max epoch is sensible)
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs
            )
        else:
            self.scheduler = scheduler
        
        # Store validation targets for metrics calculation
        self.y_val_numpy = y_val.numpy().flatten()
    
    def train(self):
        """
        Train the model with early stopping and scheduler
        
        Returns:
            Trained model and dictionary of training history
        """
        # Initialize tracking variables
        best_val_loss = float('inf')
        early_stopping_counter = 0
        history = {
            'train_loss': [],
            'val_loss': [],
            'f1_scores': []
        }
        
        # Store the best model
        best_model = None
        
        # Start training loop
        for epoch in range(self.epochs):
            print(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training phase
            self.model.train()
            train_losses = []
            total_batches = len(self.train_dataloader) # For progress bar during training batches
            
            for i, (inputs, targets) in enumerate(self.train_dataloader):
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Backward pass + optimize
                loss.backward()
                self.optimizer.step()
                
                # Store batch loss & print batch loss
                train_losses.append(loss.item())
                print(f"Batch {i+1}/{total_batches}, Training Loss: {loss.item():.4f}", end='\r')
            
            # Calculate average training loss for the epoch
            avg_train_loss = np.mean(train_losses)
            history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_losses = []
            val_predictions = []
            
            with torch.no_grad():
                # Set up progress bar for validation batches
                total_val_batches = len(self.val_dataloader)
                
                for i, (inputs, targets) in enumerate(self.val_dataloader):
                    # Move data to device
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, targets)
                    
                    # Store batch loss
                    val_losses.append(loss.item())
                    
                    # Store predictions for F1 score calculation
                    predictions = torch.sigmoid(outputs).cpu().numpy() > 0.5
                    val_predictions.extend(predictions.flatten())
                    
                    # Print batch progress
                    print(f"Validation Batch {i+1}/{total_val_batches}, Validation Loss: {loss.item():.4f}", end='\r')
            
            # Calculate average validation loss
            avg_val_loss = np.mean(val_losses)
            history['val_loss'].append(avg_val_loss)
            
            # Calculate F1 score
            f1 = f1_score(self.y_val_numpy, val_predictions, average='macro')
            history['f1_scores'].append(f1)
            
            # Print epoch results
            print(f"""Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F1 Macro: {f1:.4f}""")
            
            # decay learning rate
            self.scheduler.step()
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                
                # Save the best model
                best_model = copy.deepcopy(self.model)
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"Model saved to {self.model_save_path}")
            else:
                early_stopping_counter += 1
                print(f"Early stopping counter: {early_stopping_counter}/{self.early_stopping_patience}")
                
                if early_stopping_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        print("Training completed!")
        return best_model, history
    
    def evaluate(self, x_test, y_test):
        """
        Evaluate the trained model on test data
        
        inputs:
            x_test: features
            y_test: binary targets
            
        Returns:
            Dict with test metrics (loss, accuracy, f1_score_macro)
        """
        # Load the best model
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()
        
        # Convert data to PyTorch tensors if needed
        if not isinstance(x_test, torch.Tensor):
            x_test = torch.tensor(x_test.values if hasattr(x_test, 'values') else x_test, dtype=torch.float32)
        if not isinstance(y_test, torch.Tensor):
            y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
            
        # Create test dataloader
        test_dataloader = DataLoader(
            TensorDataset(x_test, y_test),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        test_losses = []
        test_predictions = []
        
        with torch.no_grad():
            for inputs, targets in test_dataloader:
                # Move data to device
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Store batch loss
                test_losses.append(loss.item())
                
                # Store predictions for metrics calculation
                predictions = torch.sigmoid(outputs).cpu().numpy() > 0.5
                test_predictions.extend(predictions.flatten())
        
        # Calculate metrics
        avg_test_loss = np.mean(test_losses)
        f1 = f1_score(y_test.numpy().flatten(), test_predictions, average='macro')
        accuracy = np.mean(np.array(test_predictions) == y_test.numpy().flatten())
        
        return {
            'test_loss': avg_test_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }