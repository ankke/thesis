#!/bin/bash

cp ~/relationformer/configs/road_rgb_2D.yaml ~/relationformer/configs/experiments/"$1.yaml"
sbatch <<EOT
#!/bin/sh
#SBATCH --job-name=$1
#SBATCH --output=/home/guests/alexander_berger/slurm_out/$1.out
#SBATCH --error=/home/guests/alexander_berger/slurm_out/$1.err
#SBATCH --time=5-00:00:00
#SBATCH --cpus-per-task=4  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=20GB

# load python module
ml python/anaconda3

# activate corresponding environment
conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always run this script from a clean terminal
#conda activate relationformer
source /home/guests/alexander_berger/relationformer/env/bin/activate

#python ~/relationformer/train.py --config ~/relationformer/configs/road_rgb_2D.yaml
#python ~/relationformer/models/ops/setup.py install
python /home/guests/alexander_berger/relationformer/train.py --config /home/guests/alexander_berger/relationformer/configs/experiments/$1.yaml
exit 0
EOT
