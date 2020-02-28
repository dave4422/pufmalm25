import torch


def categorical_accuracy(y, y_pred):
    """Calculates categorical accuracy.

    # Arguments:
        y_pred: Prediction probabilities or logits of shape [batch_size, num_categories]
        y: Ground truth categories. Must have shape [batch_size,]
    """
    #print(y)
    #print(torch.flatten(y))
    #print(torch.flatten(y).shape)
    #print(y_pred.shape)
    #print(torch.eq(torch.flatten(y), y_pred).sum().item())
    return torch.eq(y_pred, torch.flatten(y)).sum().item() / y_pred.shape[0]


NAMED_METRICS = {
    'categorical_accuracy': categorical_accuracy
}
