#!/bin/bash -l
#SBATCH --account=gpu_lion
#SBATCH --array=0-35
#SBATCH -o ./tjob.%A_%a.out
#SBATCH -e ./tjob.%A_%a.err
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J mdp_comparison # job dependent
# Queue (Partition):
#SBATCH --partition=gpu_lion
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=janli@mail.lorentz.leidenuniv.nl # user dependent
#
# Wall clock limit
#SBATCH --time=22:59:00

#Load some modules
module load ALICE/default
module load Python/3.10.8-GCCcore-12.2.0
module load SciPy-bundle/2023.11-gfbf-2023b
source /home/lijt/quantumNetworkInhom/rl_env/bin/activate

# adding paths for my modules
export PYTHONPATH="$PYTHONPATH:~/quantumNetworkInhom/code/src"
export PYTHONPATH="$PYTHONPATH:~/quantumNetworkInhom/code/src/environments"
export PYTHONPATH="$PYTHONPATH:~/quantumNetworkInhom/code/src/TrainSim"

# Run the program:
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_  
echo "Total number of tasks: " $SLURM_ARRAY_TASK_COUNT
N_idx=$SLURM_ARRAY_TASK_COUNT

python DataGen/SwapAsapSimHpc.py -idx $SLURM_ARRAY_TASK_ID -N_idx $N_idx -sim 1 -sim_eps "5e3"