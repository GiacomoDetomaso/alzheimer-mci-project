import torch
import numpy as np
import random
import pandas as pd
import copy

from monai.data import DataLoader
from torch.utils.data import ConcatDataset
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from IPython.display import display
from tqdm.auto import tqdm
from torch import save

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True, serialize=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.serialize = serialize
        self.best_epoch = None
        self.best_model = None
        self.best_loss = None
        self.best_optim = None
        self.best_scheduler = None
        self.counter = 0
        self.status = ""

    def __set_best(self, loss, model, optimizer, scheduler, epoch):
            self.best_loss = loss
            self.best_epoch = epoch
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_optim = copy.deepcopy(optimizer.state_dict())
            self.best_scheduler = copy.deepcopy(scheduler.state_dict())

    def __call__(self, model, optimizer, scheduler, val_loss, epoch):
        stop = False

        if self.best_loss is None:
            self.__set_best(val_loss, model, optimizer, scheduler, epoch)
        elif self.best_loss - val_loss >= self.min_delta:
            self.__set_best(val_loss, model, optimizer, scheduler, epoch)
            self.counter = 0
            self.status = f"Improvement found, counter reset to {self.counter}"
        else:
            self.counter += 1
            self.status = f"No improvement in the last {self.counter} epochs"
            if self.counter >= self.patience:
                self.status = f"Early stopping triggered after {self.counter} epochs."

                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                    optimizer.load_state_dict(self.best_optim)
                    scheduler.load_state_dict(self.best_scheduler)

                if self.serialize:
                    save(
                        obj={
                            'epoch': self.best_epoch,
                            'val_loss': self.best_loss,
                            'network': self.best_model,
                            'optimizer': self.best_optim,
                            'scheduler': self.best_scheduler
                        },
                        f=f'model_check_epoch_{self.best_epoch}.pt'
                    )

                stop = True
            
        return stop

class CustomTrainer():
    def __init__(self, model, train, validation, optimizer, loss_fn, dict_input=False, seed=42):
        self.model=model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = None
        self.current_epoch = 0
        self.dict_input = dict_input
        self.full_train = ConcatDataset([train, validation])

        # Define torch generator
        g = torch.Generator()
        g.manual_seed(seed)

        self.train_loader = DataLoader(
            train, 
            batch_size=32, 
            shuffle=True, 
            generator=g,
            worker_init_fn=self.__seed_worker
        )

        self.val_loader = DataLoader(
            validation, 
            batch_size=8, 
            shuffle=False, 
            generator=g, 
            worker_init_fn=self.__seed_worker
        )
        

    def __get_device(self):
        """
        Returns the device available on the current machine.
        Args: None
        Returns:
            device (str): name of the device available.
        """
        device = 'cpu'
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        
        return device


    def __seed_worker(self, worker_id):
        # Function to ensure reproducibility
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def __compute_running_metrics(self, label, pred, previous):  
        # Put the tensor back on the cpu and covert to numpy to compute sklearn metrics
        label = label.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        # Calculate current metrics, avoiding warning for zero divisions
        f1 = f1_score(label, pred, average='macro', zero_division=0.0)
        precision = precision_score(label, pred, average='macro', zero_division=0.0)
        recall = recall_score(label, pred, average='macro', zero_division=0.0)
        accuracy = accuracy_score(label, pred)

        return {
            'f1': previous['f1'] + f1,
            'precision': previous['precision'] + precision,
            'recall': previous['recall'] + recall,
            'accuracy': previous['accuracy'] + accuracy
        }

    def __single_epoch_train(self, data_key_name, data_label_name, use_full=False):
        # Set the model in Train mode: in this way params can be updated
        self.model.train()

        # Choose the training loader
        loader =  self.train_loader if not use_full else self.full_train_loader
      
        batches = len(loader)
        description = f'Epoch {self.current_epoch}) Training'
        progress_bar = tqdm(range(batches), leave=True, desc=description)
        scaler = torch.amp.GradScaler()

        # Init metrics
        running_loss = 0
        metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}

        for i, batch in enumerate(loader):
            # Put gradients equal to zero for every batch
            self.optimizer.zero_grad()
            labels = batch[data_label_name].to(self.__get_device())

            if self.dict_input:
                model_input = {k: v for k, v in zip(batch.keys(), batch.values()) if k in data_key_name}
            else:
                model_input = batch[data_key_name]
            
            with torch.amp.autocast(device_type=self.__get_device()):
                # Make predictions for current batch and evaluate metrics
                outputs = self.model(model_input)
                loss = self.loss_fn(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)
            metrics = self.__compute_running_metrics(labels, predictions, metrics)

            # Backpropagate the errors and parameter update
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
    
            if self.scheduler is not None:
                self.scheduler.step()
    
            running_loss += loss.item()
            progress_bar.update(1)

        # Normalize computed metrics before returning them
        avg_loss = running_loss / batches
        avg_metrics = {k: v / batches for k, v in metrics.items()}

        return avg_loss, avg_metrics
    

    def __single_epoch_validation(self, data_key_name, data_label_name):
        # The model is set to evaluation 
        self.model.eval()
      
        batches = len(self.val_loader)
        description = f'Epoch {self.current_epoch}) Validation'
        progress_bar = tqdm(range(batches), leave=True, desc=description)

        running_loss = 0
        metrics = {'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0}

        # The model is in evaluation mode: gradient calculation is disabled
        with torch.no_grad():
          for i, batch in enumerate(self.val_loader):
            labels = batch[data_label_name].to(self.__get_device())

            if self.dict_input:
                model_input = {k: v for k, v in zip(batch.keys(), batch.values()) if k in data_key_name}
            else:
                model_input = batch[data_key_name]            
            
            outputs = self.model(model_input)
            loss = self.loss_fn(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            metrics = self.__compute_running_metrics(labels, predictions, metrics)
            running_loss += loss.item()
            progress_bar.update(1)
          
        # Normalize computed metrics before returning them
        avg_loss = running_loss / batches
        avg_metrics = {k: v / batches for k, v in metrics.items()}

        return avg_loss, avg_metrics
    

    def set_scheduling(self, scheduler):
        self.scheduler = scheduler

    
    def train_model(
        self, 
        epochs, 
        early_stopping_start,
        patience,
        data_key_name='data', 
        data_label_name='label', 
        full=False, 
    ):
        out_lst = []
        self.current_epoch = 0

        early_stopper = EarlyStopping(patience=patience, min_delta=0)

        for _ in range(epochs):
            self.current_epoch += 1
            
            # Train and validate the model for the current epoch
            avg_loss_train, train_metrics  = self.__single_epoch_train(data_key_name, data_label_name, full)
            train_metrics['loss'] = avg_loss_train

            avg_loss_val, val_metrics = self.__single_epoch_validation(data_key_name, data_label_name)
            val_metrics['loss'] = avg_loss_val

            if self.current_epoch >= early_stopping_start:
                stop = early_stopper(
                    self.model, 
                    self.optimizer, 
                    self.scheduler, 
                    val_metrics['loss'], 
                    self.current_epoch
                )

                print(early_stopper.status)

                if stop:
                    dis_df = pd.DataFrame(data=[train_metrics, val_metrics], index=['Train', 'Val'])
                    display(dis_df)
                    out_lst.append(dis_df)
                    break
                        
            dis_df = pd.DataFrame(data=[train_metrics, val_metrics], index=['Train', 'Val'])

            display(dis_df)
            out_lst.append(dis_df)

        return pd.concat(out_lst)