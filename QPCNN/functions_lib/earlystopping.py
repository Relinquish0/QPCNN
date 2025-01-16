import numpy as np
import torch
import copy


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, delta=0, verbose=False, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.train_loss_min = np.Inf  # np.inf表示"正无穷"，没有确切的数值，类型为浮点型
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, train_loss, model):

        score = -train_loss

        if self.best_score is None:  # 预设置：self.best_score = None
            self.best_score = score  # 预设置：self.best_score = score == -train_loss
            self.save_checkpoint(train_loss, model)  # 保存模型并令"新的train_loss_min = 旧的train_loss"
        elif score < self.best_score + self.delta:  # 每次iteration中，score < best_score + delta
            self.counter += 1
            if self.counter >= self.patience - 5:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')  # self.trace_func即为print
            if self.counter >= self.patience:  # counter >= patience
                self.early_stop = True
        else:  # 每次iteration中，score >= best_score + delta
            self.best_score = score
            self.counter = 0  # counter置零

    def save_checkpoint(self, train_loss, model):
        '''Saves model when train loss decreases.'''
        if self.verbose:  # verbose == True
            self.trace_func(f'Train loss decreased ({self.train_loss_min:.6f} --> {train_loss:.6f}).  Saving model ...')
        best_model_wts = copy.deepcopy(model.state_dict())
        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), self.path)  # 保存模型至path路径
        self.train_loss_min = train_loss

