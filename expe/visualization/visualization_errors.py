import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import hydra

from expe.config.run_names import RUN_NAMES
from expe.visualization.visualization_functions import plot_errors_all


@hydra.main(config_path="config/", config_name="visualization.yaml")
def main(cfg: DictConfig) -> None:
    errors_dir = '/home/lise.leboudec/project/coral/xp/errors/'

    sub_from = cfg.data.sub_from
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te

    baselines = ['coral', 'dino', 'mppde', 'deeponet']

    inr_run_name = RUN_NAMES[sub_from][sub_tr]["coral"]["inr"]
    dyn_run_name = RUN_NAMES[sub_from][sub_tr]["coral"]["dyn"]
    dino_run_name = RUN_NAMES[sub_from][sub_tr]["dino"]
    mppde_run_name = RUN_NAMES[sub_from][sub_tr]["mppde"]
    deeponet_run_name = RUN_NAMES[sub_from][sub_tr]["deeponet"]
    fno_run_name = RUN_NAMES[sub_from][sub_tr]["fno"]
    if fno_run_name == "":
        fno_run_name = None
    else:
        baselines += ['fno']

    n_baselines = len(baselines)
    up_samplings = ['0', '1', '2', '4']

    # load errors
    title = f'errors_train_coral_{sub_tr}.npz'
    errors_coral_train = np.load(errors_dir + title)['arr_0']
    print("errors_coral_train.shape : ", errors_coral_train.shape)
    title = f'errors_test_coral_{sub_tr}.npz'
    errors_coral_test = np.load(errors_dir + title)['arr_0']
    title = f'errors_train_dino_{sub_tr}.npz'
    errors_dino_train = np.load(errors_dir + title)['arr_0']
    title = f'errors_test_dino_{sub_tr}.npz'
    errors_dino_test = np.load(errors_dir + title)['arr_0']
    title = f'errors_train_mppde_{sub_tr}.npz'
    errors_mppde_train = np.load(errors_dir + title)['arr_0']
    title = f'errors_test_mppde_{sub_tr}.npz'
    errors_mppde_test = np.load(errors_dir + title)['arr_0']
    title = f'errors_train_deeponet_{sub_tr}.npz'
    errors_deeponet_train = np.load(errors_dir + title)['arr_0']
    title = f'errors_test_deeponet_{sub_tr}.npz'
    errors_deeponet_test = np.load(errors_dir + title)['arr_0']
    errors_fno_train = 0
    errors_fno_test = 0
    if fno_run_name != None:
        title = f'errors_train_fno_{sub_tr}.npz'
        errors_fno_train = np.load(errors_dir + title)['arr_0']
        title = f'errors_test_fno_{sub_tr}.npz'
        errors_fno_test = np.load(errors_dir + title)['arr_0']

    plot_dir = '/home/lise.leboudec/project/coral/xp/vis/'

    # for i, up in enumerate(up_samplings):
    title = f'ns-errors-{sub_tr}-64to256-train.png'
    plot_errors_all(plot_dir, title, errors_coral_train, errors_dino_train,
                    errors_mppde_train, errors_fno_train, errors_deeponet_train, is_fno=(fno_run_name != None))
    title = f'ns-errors-{sub_tr}-64to256-test.png'
    plot_errors_all(plot_dir, title, errors_coral_test, errors_dino_test,
                    errors_mppde_test, errors_fno_test, errors_deeponet_test, is_fno=(fno_run_name != None))


if __name__ == "__main__":
    main()
