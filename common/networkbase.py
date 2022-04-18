import torch
import torch.nn as nn


__all__ = [
    'Container', 'NetworkBase'
]


class Container(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = None

    def to(self, *args, **kwargs):
        device, *_ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            device = torch.device(device)
            for module in self.children():
                if isinstance(module, Container):
                    module.to(device)
            self.device = device
        return super().to(*args, **kwargs)

    def save_model(self, path, key_filter=None, optimizer=None, alpha_optimizer=None, scaler=None):
        state_dict = self.state_dict()
        keys = list(state_dict.keys())
        for key in keys:
            if key_filter is not None and not key_filter(key):
                state_dict.pop(key)
            else:
                state_dict[key] = state_dict[key].cpu()

        state_dict = {'model': state_dict}

        if optimizer is not None:
            state_dict['optimizer'] = optimizer.state_dict()

        if alpha_optimizer is not None:
            state_dict['alpha_optimizer'] = alpha_optimizer.state_dict()

        if scaler is not None:
            state_dict['scaler'] = scaler.state_dict()

        torch.save(state_dict, path)
        return state_dict

    def load_model(self, path, strict=True):
        state_dict = torch.load(path, map_location=self.device)
        if 'optimizer' in state_dict:
            del state_dict['optimizer']

        if 'alpha_optimizer' in state_dict:
            del state_dict['alpha_optimizer']

        if 'scaler' in state_dict:
            del state_dict['scaler']

        if 'model' in state_dict:
            state_dict = state_dict['model']

        return self.load_state_dict(state_dict, strict=strict)


NetworkBase = Container
