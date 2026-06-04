"""
Shared utilities for all experiment settings.
Covers data generation, model construction, training, and ATE estimation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import platform

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.multivariate_normal import MultivariateNormal

from models.Normalizers import MonotonicNormalizer
from models.Conditionners import DAGConditioner
from models.NormalizingFlowFactories import (
    buildFCNormalizingFlow_UC,
    buildFCNormalizingFlow_UC_rho_fn,
    FixedFnConditionalLogDensity_UC,
)


# ── Causal graph helpers ─────────────────────────────────────────────────────

def get_adj_matrix():
    """Fixed A → Y adjacency matrix."""
    A = torch.zeros(2, 2)
    A[1, 0] = 1.
    return A


def get_cov_matrix(rho):
    """Latent noise covariance Z_Sigma for a given rho."""
    Z = torch.eye(2)
    Z[0, 1] = rho
    Z[1, 0] = rho
    return Z


# ── Data helpers ─────────────────────────────────────────────────────────────

def split_data(data, val_frac=0.1):
    """Split tensor [N, d] into train / val. Test == val (matches original code)."""
    N = data.shape[0]
    n_val = int(val_frac * N)
    return data[:N - n_val], data[N - n_val:]


def compute_normalisation(data_train, data_val):
    """Compute mean / std over train + val, matching UCIdatasets convention."""
    combined = torch.cat([data_train, data_val], dim=0)
    return combined.mean(0), combined.std(0)


# ── Model construction ───────────────────────────────────────────────────────

def build_rho_gnf(Z_Sigma, emb_net, int_net, nb_steps, solver, l1,
                  nb_flow, data_mu, data_sigma, nb_epoch_update, device,
                  cond_in=0, cat_dims=None):
    """
    Build a rho-GNF (DAGConditioner + MonotonicNormalizer) for a 2-variable system.

    Parameters
    ----------
    Z_Sigma       : [2, 2] covariance of the latent noise (encodes rho)
    emb_net       : list of ints, e.g. [20, 15, 10] — last entry is the embedding size
    int_net       : list of ints for the UMNN integrand network
    nb_steps      : number of integration steps for MonotonicNormalizer
    solver        : "CC" or "CCParallel"
    l1            : L1 weight on the DAG adjacency matrix
    nb_flow       : number of flow steps
    data_mu       : [d] normalisation mean
    data_sigma    : [d] normalisation std
    nb_epoch_update : how often to update the augmented Lagrangian dual params
    device        : torch device string

    Note: the augmented Lagrangian DAG constraint machinery is disabled after
    construction because A_prior is a fixed true DAG (A→Y, requires_grad=False).
    The constraint is trivially satisfied from the start, and the dual-parameter
    update loop would otherwise print log(0)=-inf every nb_epoch_update epochs
    without affecting the learned parameters.
    """
    dim = 2
    conditioner_args = {
        "in_size":          dim,
        "hidden":           emb_net[:-1],
        "out_size":         emb_net[-1],
        "l1":               l1,
        "gumble_T":         0.5,
        "nb_epoch_update":  nb_epoch_update,
        "hot_encoding":     False,
        "A_prior":          get_adj_matrix().to(device),
        "Z_Sigma":          Z_Sigma.to(device),
        "cond_in":          cond_in,
    }
    normalizer_args = {
        "integrand_net": int_net,
        "cond_size":     emb_net[-1],
        "nb_steps":      nb_steps,
        "solver":        solver,
        "mu":            data_mu.to(device),
        "sigma":         data_sigma.to(device),
        "cat_dims":      cat_dims,
    }
    model = buildFCNormalizingFlow_UC(
        nb_flow, DAGConditioner, conditioner_args,
        MonotonicNormalizer, normalizer_args,
    )
    model = model.to(device)

    # Disable the augmented Lagrangian DAG constraint on all conditioners.
    # A is a fixed true DAG (requires_grad=False), so the constraint is
    # trivially satisfied — no need for the dual-parameter update loop.
    for cond in model.getConditioners():
        cond.dag_const    = torch.tensor(0.)
        cond.l1_weight    = torch.tensor(0.)
        cond.is_invertible = True

    return model


def build_rho_fn_gnf(rho_fn, emb_net, int_net, nb_steps, solver, l1,
                     nb_flow, data_mu, data_sigma, nb_epoch_update, device,
                     cat_dims=None):
    """
    Build a rho-GNF whose latent log-density uses rho_fn(X_col) instead of a fixed rho.

    rho_fn  : callable (Tensor [B, 1]) -> Tensor [B, 1], values in (-1, 1).
              Receives the X column extracted from each training batch.

    During training, pass rho_x_col=<column_index> to train_model so that X is
    extracted from each batch and forwarded to the log-density.

    The DAGConditioner's Z_Sigma is set to identity — it is only used internally
    for the DAG constraint machinery; the actual NLL uses rho_fn.
    """
    dim = 2
    conditioner_args = {
        "in_size":          dim,
        "hidden":           emb_net[:-1],
        "out_size":         emb_net[-1],
        "l1":               l1,
        "gumble_T":         0.5,
        "nb_epoch_update":  nb_epoch_update,
        "hot_encoding":     False,
        "A_prior":          get_adj_matrix().to(device),
        "Z_Sigma":          torch.eye(dim).to(device),
        "cond_in":          0,
    }
    normalizer_args = {
        "integrand_net": int_net,
        "cond_size":     emb_net[-1],
        "nb_steps":      nb_steps,
        "solver":        solver,
        "mu":            data_mu.to(device),
        "sigma":         data_sigma.to(device),
        "cat_dims":      cat_dims,
    }
    model = buildFCNormalizingFlow_UC_rho_fn(
        nb_flow, DAGConditioner, conditioner_args,
        MonotonicNormalizer, normalizer_args,
        rho_fn=rho_fn,
    )
    model = model.to(device)

    for cond in model.getConditioners():
        cond.dag_const     = torch.tensor(0.)
        cond.l1_weight     = torch.tensor(0.)
        cond.is_invertible = True

    return model


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(model, data_train, data_val,
                nb_epoch, b_size, nb_steps, learning_rate, nb_estop, device,
                context_train=None, context_val=None,
                rho_x_col=None,
                rho_context_train=None, rho_context_val=None):
    """
    Train a rho-GNF. Returns (trained_model, val_loss_history).

    Early stopping is based on validation negative log-likelihood.
    The augmented Lagrangian DAG parameters are updated via model.step() each epoch,
    matching the original training protocol.

    If context_train / context_val are provided (shape [N, ctx_dim]), they are
    passed as conditioning context to the model on every forward call (V2 variant).

    If rho_x_col is set to a column index, that column is extracted from the data
    batch and passed as context to model.loss / z_log_density.  Use this with
    build_rho_fn_gnf so the fixed rho_fn receives the treatment variable X.

    If rho_context_train / rho_context_val are provided (shape [N, ctx_dim]),
    they are passed directly to model.loss as the rho context, independently of
    the model data.  Use this when the confounder X is not a column of the model
    data (e.g. data is (A, Y) but X must be supplied separately for rho_fn).
    rho_context takes priority over rho_x_col when both are set.
    """
    workers = 0 if platform.system() == "Windows" else 4
    has_ctx     = context_train is not None
    has_rho_ext = rho_context_train is not None

    def _make_loader(tensors, batch_size, shuffle):
        return DataLoader(
            TensorDataset(*[t.float() for t in tensors]),
            batch_size=batch_size, shuffle=shuffle,
            num_workers=workers, drop_last=False,
        )

    if has_ctx and has_rho_ext:
        l_trn = _make_loader([data_train, context_train, rho_context_train], b_size, True)
        l_val = _make_loader([data_val,   context_val,   rho_context_val],   len(data_val), False)
    elif has_ctx:
        l_trn = _make_loader([data_train, context_train], b_size, True)
        l_val = _make_loader([data_val,   context_val],   len(data_val), False)
    elif has_rho_ext:
        l_trn = _make_loader([data_train, rho_context_train], b_size, True)
        l_val = _make_loader([data_val,   rho_context_val],   len(data_val), False)
    else:
        l_trn = _make_loader([data_train], b_size, True)
        l_val = _make_loader([data_val],   len(data_val), False)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    best_val_loss = np.inf
    n_estop = 0
    val_loss_history = []

    for epoch in range(1, nb_epoch + 1):
        model.train()

        # Enforce DAG constraint at start of each epoch
        with torch.no_grad():
            for cond in model.getConditioners():
                cond.constrainA(zero_threshold=0.)

        ll_tot = 0.
        for i, batch in enumerate(l_trn):
            if has_ctx and has_rho_ext:
                cur_x, cur_ctx, cur_rho_ext = [b.to(device) for b in batch]
            elif has_ctx:
                cur_x, cur_ctx = batch[0].to(device), batch[1].to(device)
                cur_rho_ext = None
            elif has_rho_ext:
                cur_x, cur_rho_ext = batch[0].to(device), batch[1].to(device)
                cur_ctx = None
            else:
                cur_x = batch[0].to(device)
                cur_ctx, cur_rho_ext = None, None

            # Randomise integration steps slightly (matches original)
            for norm in model.getNormalizers():
                norm.nb_steps = nb_steps + torch.randint(0, 10, [1]).item()

            z, jac = model(cur_x, context=cur_ctx)

            if has_rho_ext:
                rho_ctx = cur_rho_ext
            elif rho_x_col is not None:
                rho_ctx = cur_x[:, rho_x_col:rho_x_col+1]
            else:
                rho_ctx = None

            loss = model.loss(z, jac, context=rho_ctx)

            if math.isnan(loss.item()) or math.isinf(loss.abs().item()):
                print(f"  [epoch {epoch}] NaN/Inf loss — stopping run.")
                return model, val_loss_history

            ll_tot += loss.detach()
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        ll_tot /= i + 1
        # Update augmented Lagrangian dual params for DAG constraint
        model.step(epoch, ll_tot)

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            for norm in model.getNormalizers():
                norm.nb_steps = nb_steps + 20
            ll_val = 0.
            for i, batch in enumerate(l_val):
                if has_ctx and has_rho_ext:
                    cur_x, cur_ctx, cur_rho_ext = [b.to(device) for b in batch]
                elif has_ctx:
                    cur_x, cur_ctx = batch[0].to(device), batch[1].to(device)
                    cur_rho_ext = None
                elif has_rho_ext:
                    cur_x, cur_rho_ext = batch[0].to(device), batch[1].to(device)
                    cur_ctx = None
                else:
                    cur_x = batch[0].to(device)
                    cur_ctx, cur_rho_ext = None, None

                z, jac = model(cur_x, context=cur_ctx)

                if has_rho_ext:
                    rho_ctx_val = cur_rho_ext
                elif rho_x_col is not None:
                    rho_ctx_val = cur_x[:, rho_x_col:rho_x_col+1]
                else:
                    rho_ctx_val = None

                if rho_ctx_val is not None:
                    ll_val += (model.z_log_density(z, rho_ctx_val) + jac).mean().item()
                else:
                    ll_val += (model.z_log_density(z) + jac).mean().item()
            ll_val /= i + 1

        neg_ll_val = -ll_val
        val_loss_history.append(neg_ll_val)

        if neg_ll_val < best_val_loss:
            best_val_loss = neg_ll_val
            n_estop = 0
        else:
            n_estop += 1

        if n_estop >= nb_estop:
            break

    return model, val_loss_history


# ── ATE estimation ───────────────────────────────────────────────────────────

def estimate_ate(model, Z_Sigma, n_mce_samples, mce_b_size, device,
                 treatment_dim=0, treatment_vals=(0., 1.),
                 context_samples=None):
    """
    Monte Carlo ATE estimation via flow inversion.

    Samples z from N(0, Z_Sigma), fixes the treatment dimension to each value in
    treatment_vals, inverts the flow, and returns:
        E[Y | do(A = treatment_vals[1])] - E[Y | do(A = treatment_vals[0])]

    If context_samples is provided [N, ctx_dim], n_mce_samples contexts are drawn
    with replacement and paired with z draws to marginalise over the context
    distribution (V2 variant):
        E_{X, z}[Y | do(A = a), X]

    This matches the computation in ToySimulatedContinuous.py lines 452-456.
    """
    dim = Z_Sigma.shape[0]
    n_treatments = len(treatment_vals)

    # Sample shared noise draws, then pin the treatment dimension
    z_do = MultivariateNormal(torch.zeros(dim), Z_Sigma).sample([n_mce_samples])

    # [n_treatments, n_mce_samples, dim]
    z_do_n = z_do.unsqueeze(0).expand(n_treatments, -1, -1).clone()
    for t, val in enumerate(treatment_vals):
        z_do_n[t, :, treatment_dim] = val

    # Flatten to [n_treatments * n_mce_samples, dim]
    z_do_n_flat = z_do_n.reshape(-1, dim).to(device)

    # Prepare context: sample with replacement, repeat identically across treatments
    # so that each (z_i, X_i) pair is marginalised over the same X distribution.
    ctx_flat = None
    if context_samples is not None:
        idx = torch.randint(0, len(context_samples), (n_mce_samples,))
        ctx = context_samples[idx].float()                              # [n_mce, ctx_dim]
        ctx_flat = ctx.unsqueeze(0).expand(n_treatments, -1, -1)\
                      .reshape(-1, ctx.shape[-1]).to(device)            # [n_treatments*n_mce, ctx_dim]

    # Invert in batches
    x_inv = torch.zeros_like(z_do_n_flat)
    l_z = DataLoader(
        TensorDataset(z_do_n_flat),
        batch_size=mce_b_size, shuffle=False,
        num_workers=0,
    )

    model.eval()
    offset = 0
    with torch.no_grad():
        for (z_batch,) in l_z:
            # do_val: the treatment column (already set to the intervention value)
            do_val    = z_batch[:, treatment_dim: treatment_dim + 1]
            ctx_batch = ctx_flat[offset: offset + len(z_batch)] if ctx_flat is not None else None
            x_batch   = model.invert(z_batch, context=ctx_batch, do_idx=[treatment_dim], do_val=do_val)
            x_inv[offset: offset + len(z_batch)] = x_batch
            offset += len(z_batch)

    # [n_treatments, n_mce_samples, dim] → mean over samples
    mean_outcomes = x_inv.view(n_treatments, n_mce_samples, dim).mean(1).cpu().numpy()

    # Y is the last dimension
    ate = float(mean_outcomes[1, -1] - mean_outcomes[0, -1])
    return ate


def sample_interventional_y(model, Z_Sigma, n_mce_samples, mce_b_size, device,
                             treatment_dim=0, treatment_vals=(0., 1.)):
    """
    Like estimate_ate but returns the full Y sample arrays for each treatment,
    not just the scalar mean difference.

    Returns
    -------
    list of np.ndarray, one per treatment_val, each shape [n_mce_samples].
    """
    dim = Z_Sigma.shape[0]
    n_treatments = len(treatment_vals)

    z_do   = MultivariateNormal(torch.zeros(dim), Z_Sigma).sample([n_mce_samples])
    z_do_n = z_do.unsqueeze(0).expand(n_treatments, -1, -1).clone()
    for t, val in enumerate(treatment_vals):
        z_do_n[t, :, treatment_dim] = val

    z_do_n_flat = z_do_n.reshape(-1, dim).to(device)
    x_inv = torch.zeros_like(z_do_n_flat)

    l_z = DataLoader(
        TensorDataset(z_do_n_flat),
        batch_size=mce_b_size, shuffle=False, num_workers=0,
    )

    model.eval()
    offset = 0
    with torch.no_grad():
        for (z_batch,) in l_z:
            do_val  = z_batch[:, treatment_dim: treatment_dim + 1]
            x_batch = model.invert(z_batch, do_idx=[treatment_dim], do_val=do_val)
            x_inv[offset: offset + len(z_batch)] = x_batch
            offset += len(z_batch)

    # [n_treatments, n_mce_samples, dim] → Y is the last dim
    split = x_inv.view(n_treatments, n_mce_samples, dim).cpu().numpy()
    return [split[t, :, -1] for t in range(n_treatments)]
