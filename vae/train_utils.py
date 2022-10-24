import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.cuda.amp as amp

from torch.nn.modules.loss import _Loss
from tqdm import tqdm
from utils import AverageValueMeter


class Epoch:
    def __init__(self, model, loss, stage_name, device="cpu", verbose=True):
        self.model = model
        self.loss = loss
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)


    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()

        disable_pbar = False
        if isinstance(self.device, int):
            if self.device not in [-1, 0]:
                disable_pbar = True

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose) or disable_pbar,
        ) as iterator:
            for x in iterator:
                x = x.to(self.device)
                loss, y_pred = self.batch_update(x)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {type(self.loss).__name__: loss_meter.mean}
                logs.update(loss_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss, optimizer,
                 grad_scaler: amp.grad_scaler.GradScaler, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.grad_scaler = grad_scaler

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x):
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            output, mu, logvar = self.model.forward(x)
            loss = self.loss(output, x, mu, logvar)

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return loss, output


class BetaVAELoss(nn.Module):
    __name__ = 'BetaVAELoss'

    def __init__(self, beta=3) -> None:
        super().__init__()

        self.beta = beta

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # divide total loss by batch size
        return (recon_loss + self.beta * kl_diverge) / x.size(0)
