import torch
def global_pointer_f1_score(y_true, y_pred):
    y_pred = torch.greater(y_pred, 0)
    return torch.sum(y_true * y_pred),torch.sum(y_true + y_pred)
# def global_pointer_rel_f1_score(y_head_true,y_head_pred,y_tail_true,y_tail_pred):
