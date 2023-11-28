import torch

from fno.fno_model import FNO2d
from deeponet.coral.deeponet_model import DeepONet


def load_fno(root_dir, fno_run_name):
    checkpoint = torch.load(root_dir / "fno" / f"{fno_run_name}.pt")
    cfg = checkpoint['cfg']
    modes = cfg.fno.modes
    width = cfg.fno.width
    fno = FNO2d(modes, modes, width)
    fno.load_state_dict(checkpoint['fno'])

    print("model logged at epoch : ", checkpoint['epoch'])

    return fno


def load_deeponet(root_dir, deeponet_run_name, dataset_name):
    checkpoint = torch.load(root_dir / "deeponet" /
                            f"{deeponet_run_name}_tr.pt")
    net_dyn_params = checkpoint['deeponet_params']

    deeponet = DeepONet(**net_dyn_params, logger=None,
                        input_dataset=dataset_name)
    deeponet.load_state_dict(checkpoint['deeponet_state_dict'])

    print("model logged at epoch : ", checkpoint['epoch'])

    return deeponet
