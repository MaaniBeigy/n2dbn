"""
evaluation/deep_baselines.py
============================
Two of the benchmark's own deep forecasters, trained one resolution at a time.

Why they are trained per resolution
-----------------------------------
A forecaster of this family reads a context window at one sampling rate and
predicts a horizon at that same rate.  Asked for the hourly, daily and weekly
view of the same process it produces three independent answers, and nothing in
any of them constrains the hours of a day to sum to that day.  Training one
model per resolution is what the comparison needs, because it is the only way to
give these models a cross-scale forecast at all, and the gap between their own
three answers is what the coherence column reads.

The two models
--------------
`DLinear` decomposes the context into a moving-average trend and the residual
seasonal part and puts one linear map on each, which is the published design.
`PatchTST` cuts the context into overlapping patches, embeds each, runs a small
transformer encoder over the patch sequence and flattens to the horizon.  Both
are channel-independent, which is the mode their own papers report.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:  # pragma: no cover - exercised only where torch is installed
    import torch
    from torch import nn

    _TORCH = True
except ImportError:  # pragma: no cover - the import guard itself is the branch
    _TORCH = False
    torch = None  # type: ignore[assignment]
    nn = object  # type: ignore[assignment]


def torch_available() -> bool:
    """Whether the deep baselines can run in this environment."""
    return _TORCH


def windows(
    series: np.ndarray, context: int, horizon: int, stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Every context and horizon pair a training series supplies.

    Args:
        series: `(T, L)` values at one resolution.
        context: epochs the model reads.
        horizon: epochs it predicts.
        stride: epochs between consecutive windows.

    Returns:
        An `(n, context, L)` array of contexts and an `(n, horizon, L)` array of
        targets.
    """
    series = np.asarray(series, dtype=np.float32)
    last = series.shape[0] - context - horizon
    starts = np.arange(0, last + 1, stride)
    ctx = np.stack([series[s : s + context] for s in starts])
    tgt = np.stack([series[s + context : s + context + horizon] for s in starts])
    return ctx, tgt


if _TORCH:

    class DLinear(nn.Module):
        """The decomposition-linear forecaster, one linear map per component."""

        def __init__(self, context: int, horizon: int, kernel: int = 25) -> None:
            super().__init__()
            self.kernel = kernel
            self.trend = nn.Linear(context, horizon)
            self.seasonal = nn.Linear(context, horizon)

        def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """The moving-average trend of `x` and what is left of it."""
            pad = self.kernel // 2
            padded = torch.nn.functional.pad(x, (pad, pad), mode="replicate")
            trend = torch.nn.functional.avg_pool1d(
                padded, self.kernel, stride=1, padding=0
            )
            trend = trend[..., : x.shape[-1]]
            return trend, x - trend

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict the horizon from an `(n, context, L)` context."""
            x = x.permute(0, 2, 1)
            trend, seasonal = self.decompose(x)
            out = self.trend(trend) + self.seasonal(seasonal)
            return out.permute(0, 2, 1)

    class PatchTST(nn.Module):
        """A patch transformer encoder with a flatten head, channel independent."""

        def __init__(
            self,
            context: int,
            horizon: int,
            patch: int = 16,
            stride: int = 8,
            d_model: int = 64,
            heads: int = 4,
            layers: int = 2,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            # A short context cannot be cut into patches wider than itself, and
            # the coarsest scale of this ladder is a few epochs long.
            patch = max(1, min(patch, context))
            stride = max(1, min(stride, patch))
            self.patch, self.stride = patch, stride
            n_patches = max((context - patch) // stride + 1, 1)
            self.embed = nn.Linear(patch, d_model)
            self.position = nn.Parameter(torch.zeros(1, n_patches, d_model))
            encoder = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder, num_layers=layers)
            self.head = nn.Linear(n_patches * d_model, horizon)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Predict the horizon from an `(n, context, L)` context."""
            n, context, L = x.shape
            x = x.permute(0, 2, 1).reshape(n * L, context)
            patches = x.unfold(dimension=-1, size=self.patch, step=self.stride)
            h = self.encoder(self.embed(patches) + self.position)
            out = self.head(h.reshape(h.shape[0], -1))
            return out.reshape(n, L, -1).permute(0, 2, 1)

    def train_model(
        model: "nn.Module",
        ctx: np.ndarray,
        tgt: np.ndarray,
        epochs: int = 30,
        batch: int = 128,
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 0,
    ) -> "nn.Module":
        """
        Fit one model on one resolution's windows, under the benchmark's loss.

        Args:
            model: the module to fit.
            ctx: `(n, context, L)` contexts.
            tgt: `(n, horizon, L)` targets.
            epochs: passes over the windows.
            batch: windows per step.
            lr: Adam learning rate.
            device: where to run.
            seed: fixes the initialization and the shuffling.

        Returns:
            The fitted model, in evaluation mode.
        """
        torch.manual_seed(seed)
        model = model.to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        X = torch.from_numpy(np.asarray(ctx, dtype=np.float32)).to(device)
        Y = torch.from_numpy(np.asarray(tgt, dtype=np.float32)).to(device)
        n = X.shape[0]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for _ in range(epochs):
            order = torch.randperm(n, generator=generator).to(device)
            for i in range(0, n, batch):
                idx = order[i : i + batch]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(model(X[idx]), Y[idx])
                loss.backward()
                opt.step()
        return model.eval()

    def predict(model: "nn.Module", ctx: np.ndarray, device: str = "cpu") -> np.ndarray:
        """
        The model's forecast of one context window.

        Args:
            model: a fitted module.
            ctx: `(n, context, L)` contexts.
            device: where to run.

        Returns:
            An `(n, horizon, L)` array.
        """
        with torch.no_grad():
            X = torch.from_numpy(np.asarray(ctx, dtype=np.float32)).to(device)
            return model(X).cpu().numpy()
