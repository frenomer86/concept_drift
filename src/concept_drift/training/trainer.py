from __future__ import annotations

import copy
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from concept_drift.models.dart_agil import DARTAGILNet, kl_divergence, adversarial_perturbation


class DARTAGILTrainer:
    def __init__(self, input_dim: int, n_classes: int, cfg: dict, device: str):
        self.cfg = cfg
        self.device = device
        self.model = DARTAGILNet(input_dim, cfg["latent_dim"], n_classes).to(device)
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
        )

    def _run_epoch(self, loader: DataLoader, train: bool):
        self.model.train(train)
        total_loss = 0.0
        total_n = 0
        for xb, yb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            if train:
                self.opt.zero_grad()

            logits, mu, logvar, _ = self.model(xb)
            ce = F.cross_entropy(logits, yb)
            kld = kl_divergence(mu, logvar)

            delta = adversarial_perturbation(self.model, xb, yb, self.cfg["delta_eps"], self.cfg["adv_steps"]) if train else torch.zeros_like(xb)
            adv_logits, _, _, _ = self.model(xb + delta)
            adv_loss = F.cross_entropy(adv_logits, yb)
            loss = ce + 0.1 * kld + self.cfg["lambda_agil"] * adv_loss

            if train:
                loss.backward()
                self.opt.step()

            total_loss += float(loss.item()) * xb.size(0)
            total_n += xb.size(0)
        return total_loss / max(total_n, 1)

    def fit(self, X_train, y_train, X_val, y_val):
        train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
                                  batch_size=self.cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
                                batch_size=self.cfg["batch_size"], shuffle=False)

        best = {"loss": float("inf"), "state": None, "epoch": -1}
        patience = 0
        history = []
        for epoch in range(self.cfg["epochs"]):
            train_loss = self._run_epoch(train_loader, train=True)
            val_loss = self._run_epoch(val_loader, train=False)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best["loss"]:
                best = {"loss": val_loss, "state": copy.deepcopy(self.model.state_dict()), "epoch": epoch}
                patience = 0
            else:
                patience += 1
            if patience >= self.cfg["early_stopping_patience"]:
                break

        if best["state"] is not None:
            self.model.load_state_dict(best["state"])
        return history

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        xb = torch.tensor(X, dtype=torch.float32, device=self.device)
        logits, _, _, z = self.model(xb)
        prob = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(prob, axis=1)
        return pred, prob, z.cpu().numpy()
