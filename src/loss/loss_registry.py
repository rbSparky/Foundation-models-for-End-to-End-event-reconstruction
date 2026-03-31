from torch import nn

from .conservation_loss import ConservationLoss, StableReconstructionLoss


LOSS_REGISTRY = {
    "conservation_loss": ConservationLoss,
    "stable_reconstruction_loss": StableReconstructionLoss,
    "cross_entropy_loss": nn.CrossEntropyLoss,
    "bce_with_logits_loss": nn.BCEWithLogitsLoss,
    "mse_loss": nn.MSELoss,
}
