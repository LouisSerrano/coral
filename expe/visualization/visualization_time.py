import matplotlib.pyplot as plt
import os
import json

times_dir = '/home/lise.leboudec/project/coral/expe/config/'

with open(times_dir + 'times.json', 'r') as f:
    TIMES = json.load(f)
with open(times_dir + 'times_up.json', 'r') as f:
    TIMES_UP = json.load(f)

for mode in ['train', 'test']:
    times_coral = TIMES["coral"][mode]
    times_dino = TIMES["dino"][mode]
    times_deeponet = TIMES["deeponet"][mode]
    times_fno = TIMES["fno"][mode]
    times_mppde = TIMES["mppde"][mode]

    times_coral_up = TIMES_UP["coral"][mode]
    times_dino_up = TIMES_UP["dino"][mode]
    times_deeponet_up = TIMES_UP["deeponet"][mode]
    times_fno_up = TIMES_UP["fno"][mode]
    times_mppde_up = TIMES_UP["mppde"][mode]

    baselines = ["coral", "dino", "mppde", "deeponet", "fno"]

    plt.plot(*zip(*sorted(times_coral.items())))
    plt.plot(*zip(*sorted(times_dino.items())))
    plt.plot(*zip(*sorted(times_mppde.items())))
    plt.plot(*zip(*sorted(times_deeponet.items())))
    plt.plot(*zip(*sorted(times_fno.items())), marker="x", markersize=10)
    plt.legend(baselines, ncol=5, prop={'size': 9})
    plt.yscale("log")
    plt.ylim(1e-2, 10)
    plt.title('Inference time wrt the subsampling rate.')
    plot_dir = '/home/lise.leboudec/project/coral/xp/vis/'
    title = f'times_{mode}.png'
    plt.savefig(os.path.join(plot_dir, title))
    plt.clf()
    plt.close()

    plt.plot(*zip(*sorted(times_coral_up.items())))
    plt.plot(*zip(*sorted(times_dino_up.items())))
    plt.plot(*zip(*sorted(times_mppde_up.items())))
    plt.plot(*zip(*sorted(times_deeponet_up.items())))
    plt.plot(*zip(*sorted(times_fno_up.items())))
    plt.legend(baselines, ncol=5, prop={'size': 9})
    plt.yscale("log")
    plt.ylim(1e-2, 10)
    plt.title("Inference time wrt to up-sampling rate.")
    plot_dir = '/home/lise.leboudec/project/coral/xp/vis/'
    title = f'times_up_{mode}.png'
    plt.savefig(os.path.join(plot_dir, title))
    plt.clf()
    plt.close()
