#!/bin/bash
#
#SBATCH --job-name=charrrr
#SBATCH --gres=gpu:1
#SBATCH --time=47:00:00
#SBATCH --mem=15GB
#SBATCH --output=outputs/rq_train_%A.out
#SBATCH --error=outputs/rq_train_%A.err

module purge
module load python3/intel/3.5.3 pytorch/python3.5/0.2.0_3 torchvision/python3.5/0.1.9
python3 -m pip install comet_ml --user

cd /scratch/jmw784/capstone/deep-cancer/

echo 'argument 1'
echo $1
echo 'argument 2'
echo $2

python3 -u train.py $1 --experiment $2 > logs/$2.log
