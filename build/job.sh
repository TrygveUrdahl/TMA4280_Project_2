#!/bin/sh
#SBATCH --partition=WORKQ
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --mem=12000
#SBATCH --job-name="poissontest"
#SBATCH --output=test.out
#SBATCH --mail-user=trygveur@stud.ntnu.no
#SBATCH --mail-type=END

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "The job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
module load GCC OpenMPI CMake

cmake ../src
make
mpirun hostname
mpirun -np ${SLURM_JOB_NUM_NODES} ./poisson 8192 0 ${SLURM_CPUS_ON_NODE}

uname -a
