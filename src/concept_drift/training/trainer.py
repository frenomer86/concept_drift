from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from tqdm import tqdm

from concept_drift.models.dart_agil import kl_divergence, pgd_attack


@dataclass
class TrainOutput:
    best_val_loss: float
    best_epoch: int


def run_epoch(model, loader, optimizer, device, cfg, train: bool):
    model.train(train)
    total_loss = 0.0
    count = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        logits, mu, logvar, _ = model(xb)
        bce = F.binary_cross_entropy_with_logits(logits, yb.float())
        kl = kl_divergence(mu, logvar)
        loss = bce + cfg["training"]["kl_weight"] * kl

        if cfg.get("variant", "full") != "no_agil":
            x_adv = pgd_attack(
                model,
                xb,
                yb,
                epsilon=cfg["agil"]["epsilon"],
                steps=cfg["agil"]["steps"],
                step_size=cfg["agil"]["step_size"],
            )
            logits_adv, _, _, _ = model(x_adv)
            adv_loss = F.binary_cross_entropy_with_logits(logits_adv, yb.float())
            loss = loss + cfg["training"]["agil_weight"] * adv_loss

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()

        bs = xb.shape[0]
        total_loss += loss.item() * bs
        count += bs
    return total_loss / max(1, count)


def fit(model, train_loader, val_loader, optimizer, device, cfg, checkpoint_path):
    best_val = float("inf")
    best_epoch = -1
    patience = cfg["training"]["early_stopping_patience"]
    stale = 0

    for epoch in tqdm(range(cfg["training"]["epochs"]), desc="training"):
        _ = run_epoch(model, train_loader, optimizer, device, cfg, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, cfg, train=False)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return TrainOutput(best_val_loss=best_val, best_epoch=best_epoch)
