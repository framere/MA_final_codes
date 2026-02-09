#!/bin/sh
#SBATCH -J BLock_dav
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p test
#SBATCH --time=0-16:00:00

JULIA=/home/fmereto/julia/bin/julia

$JULIA block_dav.jl
