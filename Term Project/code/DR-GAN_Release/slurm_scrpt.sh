#!/bin/bash
#SBATCH -p palamut-cuda
#SBATCH -A iperdogan
#SBATCH --time=3-00:00:00
#SBATCH -n 1
#SBATCH --job-name=biometrics
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH -N 1 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ipiskaa@gmail.com
#SBATCH --output /truba_scratch/iperdogan/biometrics/attn-sign-out-%j.out  # send stdout to outfile
#SBATCH --error /truba_scratch/iperdogan/biometrics/attn-sign-err-%j.err  # send stderr to errfile

/truba/home/iperdogan/miniconda3/envs/tensorf/bin/python /truba/home/iperdogan/biometrics/DR-GAN_Release/main_DR_GAN.py --gpu 0 --input /truba/home/iperdogan/biometrics/biometrics_final/adam_driver/cropped_side.png --out_prefix /truba/home/iperdogan/biometrics/DR-GAN_Release/outputs/out_
~                                                  