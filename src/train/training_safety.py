"""Shared checkpoint boundaries and finite optimizer updates for local trainers."""
from pathlib import Path

import torch


def validate_resume_directory(prior_config, output, checkpoint):
    original=Path(prior_config['out']).resolve()
    if Path(output).resolve()!=original or Path(checkpoint).resolve().parent!=original:
        raise ValueError('Resume must use the original experiment directory and its checkpoint')


def optimizer_step(parameters, optimizer, scaler, max_norm=5):
    parameters=list(parameters)
    scaler.unscale_(optimizer)
    norm=torch.nn.utils.clip_grad_norm_(parameters,max_norm,error_if_nonfinite=not scaler.is_enabled())
    finite=bool(torch.isfinite(norm))
    if finite:
        scaler.step(optimizer); scaler.update()
    else:
        # Never update a model using nonfinite gradients, even with finite loss.
        scaler.update(new_scale=scaler.get_scale()*.5)
    optimizer.zero_grad(set_to_none=True)
    return finite
