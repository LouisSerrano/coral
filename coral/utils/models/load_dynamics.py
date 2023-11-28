from pathlib import Path

import numpy as np
import torch

from coral.fourier_features import ModulatedFourierFeatures
from coral.mfn import FourierNet, HyperMultiscaleBACON
from coral.siren import ModulatedSiren

NAME_TO_CLASS = {
    "siren": ModulatedSiren,
    "mfn": FourierNet,
    "bacon": HyperMultiscaleBACON,
    "fourier_features": ModulatedFourierFeatures,
}


def create_dynamics_instance(cfg, device="cuda"):
    device = torch.device(device)
    if cfg.inr.model_type == "siren":
        inr = ModulatedSiren(
            dim_in=input_dim,
            dim_hidden=cfg.inr.hidden_dim,
            dim_out=output_dim,
            num_layers=cfg.inr.depth,
            w0=cfg.inr.w0,
            w0_initial=cfg.inr.w0,
            use_bias=True,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            use_latent=cfg.inr.use_latent,
            latent_dim=cfg.inr.latent_dim,
            modulation_net_dim_hidden=cfg.inr.hypernet_width,
            modulation_net_num_layers=cfg.inr.hypernet_depth,
            last_activation=cfg.inr.last_activation,
        ).to(device)
    elif cfg.inr.model_type == "mfn":
        inr = FourierNet(
            input_dim,
            cfg.inr.hidden_dim,
            cfg.inr.latent_dim,
            output_dim,
            cfg.inr.depth,
            input_scale=cfg.inr.input_scales,
        ).to(device)

    elif cfg.inr.model_type == "bacon":
        mod_activation = (
            cfg.inr.mod_activation if cfg.inr.mod_activation != "None" else None
        )
        try:
            # if input_scales look like '0.125', '0.125', etc.
            input_scales = [float(v) for v in input_scales]
        except ValueError:
            # if input_scales look like '1./8', '1./8', etc.
            input_scales = [
                float(v.split("/")[0]) / float(v.split("/")[1]) for v in input_scales
            ]
        inr = HyperMultiscaleBACON(
            input_dim,
            cfg.inr.hidden_dim,
            output_dim,
            hidden_layers=len(input_scales) - 1,
            bias=True,
            frequency=cfg.inr.frequency,
            quantization_interval=cfg.inr.quantization_multiplier * np.pi,
            input_scales=input_scales,
            output_layers=cfg.inr.output_layers,
            reuse_filters=False,
            use_latent=True,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            latent_dim=cfg.inr.latent_dim,
            modulation_net_dim_hidden=cfg.inr.hypernet_width,
            modulation_net_num_layers=cfg.inr.hypernet_depth,
            filter_type=cfg.inr.filter_type,
            mod_activation=mod_activation,
        ).to(device)

    elif cfg.inr.model_type == "fourier_features":
        inr = ModulatedFourierFeatures(
            input_dim=input_dim,
            output_dim=output_dim,
            num_frequencies=cfg.inr.num_frequencies,
            latent_dim=cfg.inr.latent_dim,
            width=cfg.inr.hidden_dim,
            depth=cfg.inr.depth,
            modulate_scale=cfg.inr.modulate_scale,
            modulate_shift=cfg.inr.modulate_shift,
            frequency_embedding=cfg.inr.frequency_embedding,
            include_input=cfg.inr.include_input,
            scale=cfg.inr.scale,
            max_frequencies=cfg.inr.max_frequencies,
            base_frequency=cfg.inr.base_frequency,
        ).to(device)

    else:
        raise NotImplementedError(f"No corresponding class for {cfg.inr.model_type}")

    return inr


def load_dynamics_model(run_dir, run_name, data_to_encode, device="cuda"):
    run_dir = Path(run_dir)
    model_train = torch.load(run_dir / f"{data_to_encode}/{run_name}.pt")
    model_state_dict = model_train["model"]
    cfg = model_train["cfg"]

    model = create_dynamics_instance(cfg, device)
    model.load_state_dict(model_state_dict)
    model.eval()

    return model
