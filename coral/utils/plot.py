import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


def imshow(true, pred, img_path, index=0):
    mse = np.mean((pred - true) ** 2)
    psnr = -10 * np.log(mse) / np.log(10)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(true[index, ...])
    axes[0].set_title("True")

    axes[1].imshow(pred[index, ...])
    axes[1].set_title(f"Pred, mse: {mse}, psnr: {psnr}")

    plt.savefig(img_path, dpi="figure")
    plt.close(fig)


def show_1D(u, v, v_hat, img_path, index=0):
    """
    u (B, d_x)
    v (B, T, d_x)
    model
    """
    ref_image = np.concatenate((np.expand_dims(u, 1), v), 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))

    axes[0, 0].imshow(ref_image[index, :, :].T, origin="upper", cmap="viridis")
    axes[0, 0].set_ylabel("x")
    axes[0, 0].set_xlabel("t")
    axes[0, 0].set_title("True")

    pred_image = np.concatenate((np.expand_dims(u, 1), v_hat), 1)

    axes[1, 0].imshow(pred_image[index, :, :].T, origin="upper", cmap="viridis")
    axes[1, 0].set_ylabel("x")
    axes[1, 0].set_xlabel("t")
    axes[1, 0].set_title("Pred")

    axes[0, 1].plot(v_hat[index, 1, :].T, label="Pred, t=1")
    axes[0, 1].plot(v[index, 1, :].T, label="True, t=1")
    axes[0, 1].plot(v_hat[index, 15, :].T, label="Pred, t=15")
    axes[0, 1].plot(v[index, 15, :].T, label="True, t=15")
    axes[0, 1].legend()

    axes[1, 1].plot(v_hat[index, 30, :].T, label="Pred, t=30")
    axes[1, 1].plot(v[index, 30, :].T, label="True, t=30")
    axes[1, 1].plot(v_hat[index, 45, :].T, label="Pred, t=45")
    axes[1, 1].plot(v[index, 45, :].T, label="True, t=45")
    axes[1, 1].legend()

    plt.savefig(img_path, dpi="figure")
    plt.show()
    plt.close(fig)


def show_2D(u, v, u_hat, v_hat, img_path, index=0):
    """
    Expects a 4 dimensional tensor. With a batch_size of 1.
    """
    T1 = u.shape[-1]
    T2 = v.shape[-1]

    fig, axes = plt.subplots(2, T1 + T2, figsize=(60, 6))

    for i in [0, 1]:
        for j in range(0, T1):
            if i == 0:
                axes[i, j].imshow(u[index, :, :, j], origin="upper", cmap="viridis")
                axes[i, j].set_title(f"Input, Time {j}")
            if i == 1:
                axes[i, j].imshow(u_hat[index, :, :, j], origin="upper", cmap="viridis")
                axes[i, j].set_title(f"Input, Time {j}")

        for j in range(0, T2):
            if i == 0:
                axes[i, T1 + j].imshow(
                    v[index, :, :, j], origin="upper", cmap="viridis"
                )
                axes[i, T1 + j].set_title(f"True, Time {j}")
            if i == 1:
                axes[i, T1 + j].imshow(
                    v_hat[index, :, :, j], origin="upper", cmap="viridis"
                )
                axes[i, T1 + j].set_title(f"Pred Time {j}")

    plt.savefig(img_path, dpi="figure")
    plt.close(fig)


def show(imgs, preds, coords, img_path, num_examples=2):
    input_dim = coords.shape[-1]
    output_dim = imgs.shape[-1]
    regular = not ((len(preds.shape[1:-1]) == 1) & (input_dim > len(preds.shape[1:-1])))
    imgs = imgs.cpu().detach()
    preds = preds.cpu().detach()
    coords = coords.cpu().detach()
    batch_size = imgs.shape[0]
    num_examples = min(batch_size, num_examples)

    print("input dim", input_dim, "output dim", output_dim, "regular", regular)

    if regular:
        if input_dim == 1:
            fig, axs = plt.subplots(1, num_examples, squeeze=False, figsize=(60, 10))
            for i in range(num_examples):
                axs[0, i].plot(
                    np.asarray(coords[i].squeeze()),
                    np.asarray(imgs[i, ...].squeeze()),
                    marker=".",
                    markersize=1,
                )
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axs[0, i].plot(
                    np.asarray(coords[i].squeeze()),
                    np.asarray(preds[i, ...].squeeze()),
                    marker=".",
                    markersize=1,
                )
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            wandb.log({img_path: fig})

        elif input_dim >= 2:
            if output_dim == 1:
                fig, axs = plt.subplots(
                    2, num_examples, squeeze=False, figsize=(60, 10)
                )
                for i in range(num_examples):
                    axs[0, i].imshow(np.asarray(imgs[i, ...].squeeze()))
                    axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    axs[1, i].imshow(np.asarray(preds[i, ...].squeeze()))
                    axs[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            if output_dim == 2:
                fig, axs = plt.subplots(
                    4, num_examples, squeeze=False, figsize=(60, 20)
                )
                for i in range(num_examples):
                    axs[0, i].imshow(np.asarray(imgs[i, ..., 0].squeeze()))
                    axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    axs[1, i].imshow(np.asarray(preds[i, ..., 0].squeeze()))
                    axs[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    axs[2, i].imshow(np.asarray(imgs[i, ..., 1].squeeze()))
                    axs[2, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    axs[3, i].imshow(np.asarray(preds[i, ..., 1].squeeze()))
                    axs[3, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            wandb.log({img_path: fig})
    else:
        if output_dim == 1:
            fig, axs = plt.subplots(2, num_examples, squeeze=False, figsize=(60, 10))
            for i in range(num_examples):
                lims = dict(cmap="RdBu_r", vmin=imgs[i].min(), vmax=imgs[i].max())
                axs[0, i].scatter(
                    np.asarray(coords[i, :, 0]),
                    np.asarray(coords[i, :, 1]),
                    50,
                    np.asarray(imgs[i, ..., 0]),
                    edgecolor="w",
                    lw=0.1,
                    **lims,
                )
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axs[1, i].scatter(
                    np.asarray(coords[i, :, 0]),
                    np.asarray(coords[i, :, 1]),
                    50,
                    np.asarray(preds[i, ..., 0]),
                    edgecolor="w",
                    lw=0.1,
                    **lims,
                )
                axs[1, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            wandb.log({img_path: fig})

        if output_dim == 2:
            fig, axs = plt.subplots(2, 2, squeeze=False)
            for i in range(num_examples):
                lims = dict(cmap="RdBu_r", vmin=imgs[i].min(), vmax=imgs[i].max())
                axs[0, 0].scatter(
                    np.asarray(coords[i, :, 0]),
                    np.asarray(coords[i, :, 1]),
                    50,
                    c=np.asarray(imgs[i, ..., 0]),
                    edgecolor="w",
                    lw=0.1,
                    **lims,
                )
                axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axs[1, 0].scatter(
                    np.asarray(coords[i, :, 0]),
                    np.asarray(coords[i, :, 1]),
                    50,
                    c=np.asarray(preds[i, ..., 0]),
                    edgecolor="w",
                    lw=0.1,
                    **lims,
                )
                axs[1, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axs[0, 1].scatter(
                    np.asarray(coords[i, :, 0]),
                    np.asarray(coords[i, :, 1]),
                    50,
                    c=np.asarray(imgs[i, ..., 1]),
                    edgecolor="w",
                    lw=0.1,
                    **lims,
                )
                axs[0, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                axs[1, 1].scatter(
                    np.asarray(coords[i, :, 0]),
                    np.asarray(coords[i, :, 1]),
                    50,
                    c=np.asarray(preds[i, ..., 1]),
                    edgecolor="w",
                    lw=0.1,
                    **lims,
                )
                axs[1, 1].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

            wandb.log({img_path: fig})

    plt.close(fig)
