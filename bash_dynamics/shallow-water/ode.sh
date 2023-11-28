#!/bin/bash
#SBATCH --partition=funky
#SBATCH --job-name=ode
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

dataset_name='shallow-water-dino'
same_grid=False
sub_from=2
sub_tr=0.2
sub_te=0.2
seq_inter_len=20
seq_extra_len=20
batch_size=64

epochs=10000
lr=0.001
weight_decay=0
gamma_step=0.75

depth=3
width=512

teacher_forcing_init=0.99
teacher_forcing_decay=0.99
teacher_forcing_update=10
inner_steps=3

#run_dict={vorticity:prime-farpoint-3079,height:worthy-sky-3242} # 40-0 inter-extra
#run_dict={vorticity:borg-dominion-3083,height:final-blood-wine-3082} # 20-20 inter-extra sub_tr = 2
#run_dict={vorticity:sandy-rain-3594,height:solar-durian-3595} # 40 - 0 inter - extra sub_tr = 4
#run_dict={vorticity:earnest-vortex-3275,height:giddy-bee-3274} # 20 - 0 inter - extra sub_tr = 0.25
#run_dict={vorticity:sleek-sun-3604,height:tough-sunset-3603} # 20 - 0 inter - extra sub_tr = 0.05
#run_dict={vorticity:lyric-cosmos-3643,height:playful-disco-3644} # 20 - 30 inter - extra en cours
#run_dict={vorticity:exalted-snow-3881,height:likely-butterfly-3880} # 20 - 20 - sub_tr = 0.05
#run_dict={vorticity:confused-vortex-3877,height:fresh-capybara-3874} # 20 - 20 sub_tr = 4
#run_dict={vorticity:dashing-morning-3878,height:expert-galaxy-3879} # 20 - 2O sub_tr = 0.25
#run_dict={vorticity:lunar-breeze-4198,height:good-fire-4199} # 20 - 20 sub_tr = 0.0125
#run_dict={vorticity:iconic-meadow-4411,height:legendary-rain-4410} # 20 - 20 sub_tr = 0.0125 w0 = 10
#run_dict={vorticity:hardy-grass-4420,height:proud-terrain-4421} # 20 - 20 sub_tr = 0.05 w0 = 10
#run_dict={vorticity:valiant-galaxy-4417,height:different-dew-4416} # 20 - 20 sub_tr = 2 w0 = 10
#run_dict={vorticity:quiet-sun-4867,height:still-bush-4868} # 0.05 same_grid = False
run_dict={vorticity:crisp-pyramid-4866,height:proud-resonance-4869} # 0.20 same_grid=False

python3 dynamics_modeling/train.py "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "dynamics.width=$width" "dynamics.depth=$depth" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.epochs=$epochs" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "optim.batch_size=$batch_size" "optim.lr=$lr"  "dynamics.teacher_forcing_update=$teacher_forcing_update" "inr.run_dict=$run_dict"
