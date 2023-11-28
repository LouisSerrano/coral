import torch
import einops

from deeponet import AR_forward

def forward_deeponet(deeponet, batch, timestamps, device):
    (images, _, coords, idx) = batch
    # flatten spatial dims
    t = timestamps.to(device)
    ground_truth = einops.rearrange(images, 'B ... C T -> B (...) C T')
    model_input = einops.rearrange(coords, 'B ... C T -> B (...) C T')

    # permute axis for forward
    ground_truth = torch.permute(
        ground_truth, (0, 3, 1, 2)).to(device)  # [B, XY, C, T] -> [B, T, XY, C]
    model_input = torch.permute(
        model_input, (0, 3, 1, 2))[:, 0, :, :].to(device)  # [B, XY, C, T] -> -> [B, T, XY, C] -> [B, XY, C]
    # On prend que la première grille (c'est tjs la mm dans deeponet) 
    b_size, t_size, hw_size, channels = ground_truth.shape

    # t is T, model_input is B, XY, grid, ground_truth is B, T, XY, C

    model_output = AR_forward(deeponet, t, model_input, ground_truth) 

    return model_output


def forward_deeponet_up(deeponet, images, images_up, coords, coords_up, timestamps, device):
    # flatten spatial dims
    t = timestamps.to(device)
    ground_truth = einops.rearrange(images, 'B ... C T -> B (...) C T')
    ground_truth_up = einops.rearrange(images_up, 'B ... C T -> B (...) C T')
    model_input = einops.rearrange(coords, 'B ... C T -> B (...) C T')
    model_input_up = einops.rearrange(coords_up, 'B ... C T -> B (...) C T')

    # permute axis for forward
    model_input = torch.permute(
        model_input, (0, 3, 1, 2))[:, 0, :, :].to(device)  # [B, XY, C, T] -> -> [B, T, XY, C] -> [B, XY, C]
    model_input_up = torch.permute(
        model_input_up, (0, 3, 1, 2))[:, 0, :, :].to(device)  # [B, XY, C, T] -> -> [B, T, XY, C] -> [B, XY, C]
    # On prend que la première grille (c'est tjs la mm dans deeponet) 
    b_size, t_size, hw_size, channels = ground_truth.shape

    # t is T, model_input is B, XY, grid, ground_truth is B, T, XY, C
    model_output = AR_forward_up(deeponet, t, model_input, model_input_up, ground_truth, ground_truth_up) 

    return model_output

def AR_forward_up(deeponet, timestamps, coords, coords_up, ground_truth, ground_truth_up, logger=None, is_test=False):
    b_size = ground_truth.shape[0]
    
    # first AR forward on original grid
    x_trunk_in_s = coords
    x_branch = ground_truth[:, :1, :, :].squeeze()  # [32, 10, 1992, 1] vs [32, 10, 1992, 2]
    if b_size == 1:
        x_branch = x_branch.unsqueeze(0)
    if len(x_branch.shape) == 3:
        x_branch = x_branch.flatten(start_dim=1)

    # CI, on prend que le 1er pas de temps et on squeeze = B, XY, C    
    for i, t in enumerate(timestamps):
        # x_branch is B, XY
        # x_trink_in_s is B, T, XY, grid
        if len(x_branch.shape) == 3:
            x_branch = x_branch.flatten(start_dim=1)
        # X_trunk devrait etre B, XY, grid et X_branch devrait etre B, XY
        im = deeponet(x_branch, x_trunk_in_s).unsqueeze(dim=1)
        if i == 0: # manage initial condition
            if is_test:
                model_output = ground_truth.reshape(*ground_truth.shape[:2], -1, *ground_truth.shape[-1:])[:, :1, :, :]
            else:
                model_output = ground_truth[:, :1, :, :]
        else:
            model_output = torch.cat((model_output, im), dim=1)
        x_branch = im.squeeze()
        if b_size == 1:
            x_branch = x_branch.unsqueeze(0)
    # 2eme AR_forward pour up-sampling
    x_trunk_in_s = coords_up
    x_branch = ground_truth[:, :1, :, :].squeeze()  # [32, 10, 1992, 1] vs [32, 10, 1992, 2]

    if b_size == 1:
        x_branch = x_branch.unsqueeze(0)
    
    if len(x_branch.shape) == 3:
        x_branch = x_branch.flatten(start_dim=1)
    # CI, on prend que le 1er pas de temps et on squeeze = B, XY, C
    for i, t in enumerate(timestamps):
        if len(x_branch.shape) == 3:
            x_branch = x_branch.flatten(start_dim=1)
        # x_branch is B, XY
        # x_trink_in_s is B, T, XY, grid

        # X_trunk devrait etre B, XY, grid et X_branch devrait etre B, XY
        im = deeponet(x_branch, x_trunk_in_s).unsqueeze(dim=1)
        
        if i == 0: # manage initial condition
            if is_test:
                model_output_up = ground_truth_up.reshape(*ground_truth_up.shape[:2], -1, *ground_truth_up.shape[-1:])[:, :1, :, :]
            else:
                model_output_up = ground_truth_up[:, :1, :, :]
        else:
            model_output_up = torch.cat((model_output_up, im), dim=1)

        # HERE : AR on the train grid = récupérer cette frame de l'AR n1 du dessus 
        if i < (len(timestamps) - 1): # pour éviter d'aller chercher un pas de temps qui n'existe pas = pas de next x_branch puisqu'on est à la fin
            x_branch = model_output[:, i+1, :, :].squeeze() 
            if b_size == 1:
                x_branch = x_branch.unsqueeze(0)

    return model_output_up