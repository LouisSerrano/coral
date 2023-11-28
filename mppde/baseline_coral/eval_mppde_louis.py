import torch
from mppde.baseline_coral.scheduling import mppde_test_rollout


def eval_mppde(model, test_loader, ntest, sequence_length_in, sequence_length_out):
    pred_test_mse = 0
    pred_test_inter_mse = 0
    pred_test_extra_mse = 0

    for graph, idx in test_loader:
        model.eval()
        n_samples = len(idx)

        graph = graph.cuda()

        with torch.no_grad():
            u_pred = mppde_test_rollout(model, graph, bundle_size=1)
            # graph.images.shape, u_pred.shape : torch.Size([816, 1, 20]) torch.Size([816, 1, 20])
            pred_test_mse += ((u_pred - graph.images)
                              ** 2).mean() * n_samples

            pred_test_inter_mse += ((u_pred[..., :sequence_length_in] - graph.images[..., :sequence_length_in])
                                    ** 2).mean() * n_samples
            pred_test_extra_mse += ((u_pred[..., sequence_length_in: sequence_length_in+sequence_length_out] - graph.images[..., sequence_length_in: sequence_length_in+sequence_length_out])
                                    ** 2).mean() * n_samples

    # code_test_mse = code_test_mse / ntest
    pred_test_mse = pred_test_mse / ntest
    pred_test_inter_mse = pred_test_inter_mse / ntest
    pred_test_extra_mse = pred_test_extra_mse / ntest

    return pred_test_mse, pred_test_inter_mse, pred_test_extra_mse
