import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import einops


def plot_baselines(plot_dir, title, b0, b1, b4, b16, pred0, pred1, pred4, pred16, x, y, time2show, baselines):
    """
    plot baselines
    predi sont des tuples contenant 1 trajectoire 
    """
    n_baselines = len(baselines)

    fig, axs = plt.subplots(n_baselines + 1, 4, figsize=(18, 7))
    fig.suptitle(f'Results on timestep {time2show}')
    axs[0, 0].scatter(y, -x, 5, b0[0, :, 0, time2show], edgecolor="w",
                      lw=0.2,)
    axs[0, 1].imshow(b1[0, :, 0, time2show].reshape(64, 64))
    axs[0, 2].imshow(b4[0, :, 0, time2show].reshape(128, 128))
    axs[0, 3].imshow(b16[0, :, 0, time2show].reshape(256, 256))

    for i in range(n_baselines):
        axs[i+1, 0].scatter(y, -x, 5, pred0[i][0, :, 0, time2show], edgecolor="w",
                            lw=0.2,)
        axs[i+1, 1].imshow(pred1[i][0, :, 0, time2show].reshape(64, 64))
        axs[i+1, 2].imshow(pred4[i][0, :, 0, time2show].reshape(128, 128))
        # axs[i+1, 2].imshow(einops.rearrange(pred4[i][0, :, 0, time2show], '(X Y) -> X Y', X=128))
        axs[i+1, 3].imshow(pred16[i][0, :, 0,
                           time2show].reshape(256, 256))  # , cmap='BuPu')

    [axi.set_axis_off() for axi in axs.ravel()]

    cols = ['x_tr', '64x64', '128x128', '256x256']
    rows = ['true'] + baselines

    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=0, size='large')

    plt.savefig(os.path.join(plot_dir, title), bbox_inches='tight', dpi=300)


def gif_baselines(plot_dir, title, b0, b1, b4, b16, x, y):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 7))

    T = b0.shape[3]

    # ims is a list of lists, each row is a list of artists to draw in the
    # current frame; here we are just animating one artist, the image, in
    # each frame
    ims = []
    for i in range(T):
        cmap = "twilight_shifted"
        if i >= 20:
            cmap = "coolwarm"
        im1 = ax1.scatter(y, -x, 5, b0[0, :, 0, i], edgecolor="w",
                          lw=0.2, cmap=cmap, animated=True)
        im2 = ax2.imshow(b1[0, :, 0, i].reshape(
            64, 64), cmap=cmap)
        im3 = ax3.imshow(b4[0, :, 0, i].reshape(
            128, 128), cmap=cmap)
        im4 = ax4.imshow(b16[0, :, 0, i].reshape(
            256, 256), cmap=cmap)
        [axi.set_axis_off() for axi in [ax1, ax2, ax3, ax4]]

        ims.append([im1, im2, im3, im4])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    ani.save(os.path.join(plot_dir, title),
             dpi=300)  # , writer=PillowWriter(fps=25))


def plot_errors(plot_dir, title, b0, b1, b4, b16, pred0, pred1, pred4, pred16, x, y, baselines):
    """
    plot baselines
    predi sont des tuples contenant 1 trajectoire 
    """
    n_baselines = len(baselines)
    ts = torch.arange(b0.shape[3])

    fig, axs = plt.subplots(n_baselines, 4, figsize=(18, 7))
    fig.suptitle(f'Errors wrt time')

    for i in range(n_baselines):
        errors0 = ((b0 - pred0[i])**2).mean(0).mean(0).mean(0)  # B, XY, C, T
        errors1 = ((b1 - pred1[i])**2).mean(0).mean(0).mean(0)  # B, XY, C, T
        errors4 = ((b4 - pred4[i])**2).mean(0).mean(0).mean(0)  # B, XY, C, T
        errors16 = ((b16 - pred16[i]) **
                    2).mean(0).mean(0).mean(0)  # B, XY, C, T

        if n_baselines == 1:
            axs[0].plot(ts, errors0)
            axs[1].plot(ts, errors1)
            axs[2].plot(ts, errors4)
            axs[3].plot(ts, errors16)
        else:
            axs[i, 0].plot(ts, errors0)
            axs[i, 1].plot(ts, errors1)
            axs[i, 2].plot(ts, errors4)
            axs[i, 3].plot(ts, errors16)

    # [axi.set_axis_off() for axi in axs.ravel()]

    cols = ['x_tr', '64x64', '128x128', '256x256']
    rows = baselines

    if n_baselines == 1:
        for ax, col in zip(axs, cols):
            ax.set_title(col)
    else:
        for ax, col in zip(axs[0], cols):
            ax.set_title(col)

    plt.savefig(os.path.join(plot_dir, title), bbox_inches='tight', dpi=300)


def plot_errors_all(plot_dir, title, errors_coral, errors_dino, errors_mppde, errors_fno, errors_deeponet, is_fno=True):
    """
    plot baselines
    predi sont des tuples contenant 1 trajectoire 
    """
    n_baselines = 4 + is_fno
    ts = torch.arange(errors_coral.shape[1])
    up_samplings = ['0', '1', '2', '4']
    baselines = ['coral', 'dino', 'mppde', 'deeponet']
    if is_fno:
        baselines += ['fno']

    fig, axs = plt.subplots(1, 4, figsize=(20, 7))
    fig.subplots_adjust(top=1, hspace=0.3)
    suptitle = fig.suptitle(f'Errors wrt time', fontsize=18)
    suptitle.set_y(1.17)

    for i, up in enumerate(up_samplings):
        axs[i].plot(ts[1:], errors_coral[i, 1:])
        axs[i].plot(ts[1:], errors_dino[i, 1:])
        axs[i].plot(ts[1:], errors_mppde[i, 1:])
        axs[i].plot(ts[2:], errors_deeponet[i, 2:])
        errmin = np.min((errors_coral[i, 1:], errors_dino[i, 1:],
                         errors_mppde[i, 1:], errors_deeponet[i, 1:]))
        if is_fno:
            axs[i].plot(ts[1:], errors_fno[i, 1:])
            errmin = np.min((errors_coral[i, 1:], errors_dino[i, 1:],
                             errors_mppde[i, 1:], errors_deeponet[i, 1:], errors_fno[i, 1:]))
        axs[i].axvline(x=20, ls=':')
        axs[i].text(7, errmin, 'In-t')
        axs[i].text(27, errmin, 'Out-t')
        axs[i].set_yscale("log")

    cols = ['x_tr', '64x64', '128x128', '256x256']
    fig.legend(baselines, loc='upper center', bbox_to_anchor=(0.5, 1.12),
               ncol=n_baselines, prop={'size': 15})

    for ax, col in zip(axs, cols):
        ax.set_title(col)

    plt.savefig(os.path.join(plot_dir, title), bbox_inches='tight', dpi=300)


def plot_codes(plot_dir, title, codes, metrics):
    fig, axs = plt.subplots(1, 1, figsize=(14, 7))
    fig.subplots_adjust(top=1, hspace=0.3)
    suptitle = fig.suptitle(f'Errors wrt code dim', fontsize=18)

    axs.plot(codes, metrics)
    axs.set_yscale("log")

    plt.savefig(os.path.join(plot_dir, title), bbox_inches='tight', dpi=300)


def save_imshow(plot_dir, title, img, times2show, sres):
    for i, t, in enumerate(times2show):
        fig = plt.figure(figsize=(7, 7))
        plt.imshow(img[0, :, 0, t].reshape(
            (sres, sres)), cmap='twilight_shifted')
        plt.axis('off')
        plt.savefig(os.path.join(plot_dir, title +
                    f'-{t}.png'), bbox_inches='tight', pad_inches=0, dpi=300)

        plt.close()
        fig.clear()
        plt.clf()


def save_scatter(plot_dir, title, img, times2show, x, y):
    for i, t, in enumerate(times2show):
        fig = plt.figure(figsize=(7, 7))
        plt.scatter(y, -x, 5, img[0, :, 0, t], edgecolor="w",
                    lw=0.2, cmap='twilight_shifted')
        plt.axis('off')
        plt.savefig(os.path.join(plot_dir, title +
                    f'-{t}.png'), bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        fig.clear()
        plt.clf()


def plot_grid(plot_dir, title, x, y, x1, y1, sres=64):

    grid = np.zeros(x1.shape)

    for i, (xi, yi) in enumerate(zip(x1, y1)):
        grid[i] = 1 if (xi, yi) in zip(x, y) else 0
    print("grid.sum() : ", grid.sum())
    fig = plt.figure(figsize=(7, 7))
    plt.imshow(grid.reshape(sres, sres), vmin=0, vmax=1) # , cmap='twilight')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, title),
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    fig.clear()
    plt.clf()


def plot_grid2(plot_dir, title, x, y):

    fig = plt.figure(figsize=(7, 7))
    plt.scatter(y, -x, 15, np.ones(x.shape), edgecolor="w",
                lw=0.2, cmap='twilight_shifted')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, title),
                bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    fig.clear()
    plt.clf()
