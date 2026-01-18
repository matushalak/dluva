#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=Adversarial_attacks
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=04:00:00
#SBATCH --output=slurm_output_%A.out

module purge
module load 2025
module load Anaconda3/2025.06-1

# Your job starts in the directory where you call sbatch
cd $HOME/dluva/assignment3/part2
# Activate your environment
source activate dl2024
# Run your code

# Pretrained finetuning
# pretrained baseline: standard 92 / 44.4 fgsm / 8.6 pgd
# finetuning pretrained fgsm defense: standard 91 / fgsm 44.4
# finetuning pretrained pgd defense: standard 85 / pgd 38.7
srun python train.py --pretrained --train_strats standard fgsm pgd --test_crossover_defense

# Pretrained finetuning + augmentation
# pretrained baseline: standard 93 / 45.12 fgsm / 8.24 pgd
# finetuning pretrained fgsm defense + augment: standard 91 / fgsm 60.4 / pgd 30.4
# finetuning pretrained pgd defense + augment: standard 87 / fgsm 54.92 / pgd 40.8
srun python train.py --pretrained --train_strats standard fgsm pgd --augmentations --test_crossover_defense

# From scratch defenses without augmentations
# STANDARD: standard 68 / fgsm 1.7 / pgd 0
# FGSM: standard 60 / fgsm 40
# PGD: standard 49 / pgd 26.4
srun python train.py --train_strats standard fgsm pgd --test_crossover_defense

# From scratch defenses with augmentations
# STANDARD: standard 68 / 7.4 fgsm / 0.04 pgd
# FGSM: standard 53 / 37.8 fgsm
# PGD: standard 44 / 25.2 pgd
srun python train.py --train_strats standard fgsm pgd --augmentations --test_crossover_defense
