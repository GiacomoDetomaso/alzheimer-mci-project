import copy
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
