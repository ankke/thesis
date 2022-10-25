#!/bin/bash

cp ./configs/road_rgb_2D.yaml "./configs/experiments/$1.yaml"
sbatch ./test.slurm $1