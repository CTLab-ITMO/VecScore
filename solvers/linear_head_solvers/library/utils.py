import torch.nn as nn


def freeze_layer_until(m, idx):
    """Freezes layers up until specified index."""
    for i, child in enumerate(m.children()):
        if i <= idx and isinstance(child, nn.Module):
            child.eval()
            for param in child.parameters():
                param.requires_grad_(False)


def make_model(model, freeze_coef, new_layers, freeze=True):
    if freeze:
        # Set the number of initial blocks to keep frozen
        n_blocks_to_freeze = int(len(list(model.children())) * freeze_coef)
    
        # Freeze the early layers
        freeze_layer_until(model, n_blocks_to_freeze)

    # Feature extraction still uses fixed features from pre-trained weights but now allows later layers to learn
    model.train()

    # Combine the base model and new layers
    model.classifier = new_layers
    return model
