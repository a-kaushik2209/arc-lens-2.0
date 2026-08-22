"""
ARC Lens — reference training script
====================================

A real convolutional network trained on real CIFAR-10. Nothing here is
simulated: no injected NaN, no scripted failure step, no hardcoded curve.

**Why it is unstable.** The hyperparameters are deliberately aggressive in a way
a practitioner plausibly gets wrong: a high peak learning rate for this
architecture, a very short warmup, and no gradient clipping. That is the single
most common real cause of a diverged run. Whether it actually diverges — and at
which step — depends on the data order and the initialisation, so it is genuinely
not known in advance. That is the point: ARC has to *detect* the failure rather
than be told where it is.

Run it two ways to see what ARC is worth:

    ARC_MODE=baseline   interventions suppressed, telemetry only
    ARC_MODE=active     interventions applied  (default)

Both arms are seeded identically, so any divergence between them is caused by
the interventions and nothing else.
"""

import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data_cache

SEED = int(os.environ.get("ARC_DEMO_SEED", "1234"))
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = int(os.environ.get("ARC_DEMO_EPOCHS", "6"))
BATCH_SIZE = int(os.environ.get("ARC_DEMO_BATCH", "128"))
# 0.5 with SGD+momentum on a 9-layer CNN is past the edge of stability for this
# architecture. It is not an absurd number — it is the kind of value copied from
# a paper that used a different model, a different batch size and warmup.
PEAK_LR = float(os.environ.get("ARC_DEMO_LR", "0.5"))
WARMUP_STEPS = int(os.environ.get("ARC_DEMO_WARMUP", "60"))
DATA_ROOT = data_cache.cifar_root()


# ─────────────────────────────────────────────────────────────────────────────
# Model — 9 parameterised layers, deep enough for layer-wise gradient-flow
# diagnostics to be meaningful (they compare early against late quartiles).
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, cin, cout, pool=False):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.pool = pool

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return F.max_pool2d(x, 2) if self.pool else x


class DemoCNN(nn.Module):
    """VGG-style CIFAR network, 2,788,042 parameters."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64), ConvBlock(64, 64, pool=True),
            ConvBlock(64, 128), ConvBlock(128, 128, pool=True),
            ConvBlock(128, 256), ConvBlock(256, 256, pool=True),
            ConvBlock(256, 256),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_loaders():
    from torchvision import datasets, transforms

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    root = os.path.abspath(DATA_ROOT)
    data_cache.warn_if_cache_incomplete(root)
    train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)

    generator = torch.Generator().manual_seed(SEED)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                   drop_last=True, generator=generator),
        DataLoader(test, batch_size=512, shuffle=False, num_workers=0),
    )


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        out = model(x)
        loss_sum += F.cross_entropy(out, y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    model.train()
    return loss_sum / max(1, total), 100.0 * correct / max(1, total)


def main():
    train_loader, test_loader = build_loaders()

    model = DemoCNN().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=PEAK_LR, momentum=0.9, weight_decay=5e-4)

    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch

    mode = os.environ.get("ARC_MODE", "active")
    print(f"ARC Lens demo | device={DEVICE} mode={mode} seed={SEED}")
    print(f"  model=DemoCNN params={sum(p.numel() for p in model.parameters()):,}")
    print(f"  epochs={EPOCHS} batch={BATCH_SIZE} steps/epoch={steps_per_epoch} total={total_steps}")
    print(f"  optimizer=SGD(momentum=0.9) peak_lr={PEAK_LR} warmup={WARMUP_STEPS} clipping=off")
    print("  No failure is injected. Instability, if any, comes from these settings.")

    global_step = 0
    started = time.time()
    best_acc = 0.0
    final_loss, final_acc = float("nan"), 0.0

    for epoch in range(EPOCHS):
        # Optional helper that runner.py installs into builtins; harmless when
        # this script is run bare.
        #
        # The guard is a bare NameError catch rather than an inspection of
        # __builtins__, which is a module in some execution contexts and a dict
        # in others — `hasattr` and `dir()` disagree between the two, so the
        # previous check was simply always false and every metric reported
        # epoch 0. A plain name lookup works in both cases.
        try:
            arc_set_epoch(epoch)  # noqa: F821 — injected by runner.py
        except NameError:
            pass

        running = correct = seen = 0
        running_loss = 0.0

        for x, y in train_loader:
            global_step += 1

            # Linear warmup, then cosine decay. The warmup is short on purpose.
            if global_step <= WARMUP_STEPS:
                scale = global_step / max(1, WARMUP_STEPS)
            else:
                progress = (global_step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
                scale = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
            # A plain schedule that recomputes LR from the peak every step —
            # exactly the pattern that erases a naive LR intervention. ARC
            # re-asserts its own reduction on top of this (see enforce_lr).
            for group in optimizer.param_groups:
                group["lr"] = PEAK_LR * scale

            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

            value = loss.item()
            if math.isfinite(value):
                running_loss += value
                running += 1
                correct += (out.argmax(1) == y).sum().item()
                seen += y.numel()

        train_loss = running_loss / max(1, running)
        train_acc = 100.0 * correct / max(1, seen)
        val_loss, val_acc = evaluate(model, test_loader)
        best_acc = max(best_acc, val_acc)
        final_loss, final_acc = val_loss, val_acc

        print(
            f"[epoch {epoch + 1}/{EPOCHS}] step={global_step} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.4e} "
            f"elapsed={time.time() - started:.0f}s"
        )

    elapsed = time.time() - started
    print(f"Finished in {elapsed:.0f}s | best val_acc={best_acc:.2f}%")

    # Structured result line so an A/B harness can compare arms without
    # scraping the human-readable log.
    #
    # The final figures are the last epoch's, reused rather than recomputed —
    # re-running `evaluate` here produced an identical number at the cost of a
    # full extra pass over the test set.
    print(json.dumps({
        "type": "final_result",
        "mode": os.environ.get("ARC_MODE", "active"),
        "seed": SEED,
        "peak_lr": PEAK_LR,
        "epochs": EPOCHS,
        "steps": global_step,
        "best_val_acc": round(best_acc, 4),
        "final_val_acc": round(final_acc, 4),
        "final_val_loss": None if not math.isfinite(final_loss) else round(final_loss, 6),
        "wall_seconds": round(elapsed, 2),
    }), flush=True)


if __name__ == "__main__":
    main()
