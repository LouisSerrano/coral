import torch.nn as nn
import torch
import einops
import numpy as np


def eval_fno(loader, fno, sequence_length_ext, sequence_length_in, sigma, mean, n_seq):
    mse, mse_inter, mse_extra = 0, 0, 0
    for u_batch, modulations, coord_batch, idx in loader:
        spatial_res = int(np.sqrt(u_batch.shape[1]))
        u_batch = einops.rearrange(
            u_batch, 'B (X Y) C T -> B X Y C T', X=spatial_res)
        u_batch = u_batch.cuda()
        coord_batch = coord_batch.cuda()
        input_frame = u_batch[..., 0]
        batch_size = u_batch.shape[0]

        for t in range(sequence_length_ext - 1):
            target_frame = u_batch[..., t+1] # B, X, Y, C
            pred = fno(input_frame) # B, X, Y, C
            # loss = nn.MSELoss(pred.view(batch_size, -1),
            #                   target_frame.view(batch_size, -1))
            # test_l2 += loss.item()
            xx = pred * sigma.cuda() + mean.cuda()
            yy = target_frame * sigma.cuda() + mean.cuda()

            # We don't use the first timesteps to compute the extrapolation loss
            if t >= sequence_length_in:
                mse_extra += ((xx.view(batch_size, -1) -
                               yy.view(batch_size, -1))**2).mean()*batch_size
            if t < sequence_length_in:
                mse_inter += ((xx.view(batch_size, -1) -
                               yy.view(batch_size, -1))**2).mean()*batch_size
            mse += ((xx.view(batch_size, -1) -
                     yy.view(batch_size, -1))**2).mean()*batch_size
            input_frame = pred.detach()

    return mse / n_seq, mse_inter / n_seq, mse_extra / n_seq
