import torch
import torch.nn as nn

class UncertaintyLossWrapper(nn.Module):
    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion
        self.log_vars = nn.ParameterDict()
        self.initialized = False

    def initialize(self, loss_dict, weight_dict):
        device = next(iter(loss_dict.values())).device
        weight_keys = set(self.base_criterion.weight_dict.keys())
        for name in loss_dict:
            if name in weight_keys and name not in self.log_vars:
                self.log_vars[name] = nn.Parameter(torch.zeros(1, device=device))
        self.initialized = True

    def forward(self, outputs, targets, pre_outputs=None, pre_targets=None):
        loss_dict = self.base_criterion(outputs, targets, pre_outputs, pre_targets)
        weight_dict = self.base_criterion.weight_dict

        if not self.initialized:
            self.initialize(loss_dict, weight_dict)

        total_loss = 0
        weighted_losses = {}
        learned_log_vars = {}

        for name, loss in loss_dict.items():
            if name not in weight_dict:
                weighted_losses[name] = loss.detach()
                continue

            if name not in self.log_vars:
                self.log_vars[name] = nn.Parameter(torch.zeros(1, device=loss.device))

            log_var = self.log_vars[name]
            precision = torch.exp(-log_var)
            weighted_loss = precision * loss + log_var
            total_loss += weighted_loss

            weighted_losses[name] = weighted_loss.detach()
            learned_log_vars[name] = log_var.detach().item()

        return total_loss, loss_dict, weighted_losses, learned_log_vars
