"""
VAE loss: reconstruction + KL divergence.
Optionally supports LPIPS perceptual loss when lpips is installed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(
        self,
        rec_loss="l1",
        kl_weight=1.0,
        perceptual_weight=0.0,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight

        if rec_loss == "l1":
            self.rec_criterion = nn.L1Loss(reduction="mean")
        elif rec_loss == "l2":
            self.rec_criterion = nn.MSELoss(reduction="mean")
        else:
            raise ValueError(f"Unknown rec_loss: {rec_loss}")

        self.perceptual_loss = None
        if self.perceptual_weight > 0.0:
            try:
                import lpips
                self.perceptual_loss = lpips.LPIPS(net="vgg").eval()
                for p in self.perceptual_loss.parameters():
                    p.requires_grad = False
            except ImportError:
                raise ImportError(
                    "perceptual_weight > 0 requires lpips. Install with: pip install lpips"
                )

    def forward(self, inputs, reconstructions, posterior):
        rec_loss = self.rec_criterion(inputs, reconstructions)

        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        loss = rec_loss + self.kl_weight * kl_loss

        log_dict = {
            "loss": loss.detach(),
            "rec_loss": rec_loss.detach(),
            "kl_loss": kl_loss.detach(),
        }

        if self.perceptual_weight > 0.0 and self.perceptual_loss is not None:
            # LPIPS expects images in [-1, 1]
            p_loss = self.perceptual_loss(inputs, reconstructions).mean()
            loss = loss + self.perceptual_weight * p_loss
            log_dict["perceptual_loss"] = p_loss.detach()

        return loss, log_dict
