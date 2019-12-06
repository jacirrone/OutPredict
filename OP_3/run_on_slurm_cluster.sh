#!/bin/bash

#PRINCE PRINCE_SACCT=YES

#SBATCH --job-name=OP3
#SBARCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=125GB
#SBATCH --time=12:00:00

#ppn=32,mem=512GB,walltime=12:00:00
#ppn=20,mem=189GB,walltime=12:00:00
#ppn=20,mem=189GB,walltime=8:00:00
#ppn=28,mem=125GB,walltime=2:00:00
#ppn=20,mem=125GB,walltime=2:00:00
#ppn=28,mem=62GB,walltime=1:00:00
#ppn=20,mem=62GB,walltime=1:00:00
#####SBATCH --output=slurm_output_files/slurm_%j.out

#There is 60GB or 125GB memory on most of the compute node on prince.
#https://wikis.nyu.edu/display/NYUHPC/Slurm+Tutorial#SlurmTutorial-Princecomputingnodes


module purge
module load anaconda3/5.3.1
source activate op3


cd /home/jc3832/OutPredict3/OP_3

# job_name="job"
# _dir=$(pwd)
# output_dir=${job_name}-output

# if [ "$SLURM_JOBTMP" == "" ]; then
#     echo " No SLURM_JOBTMP available"
#     exit 1
# fi

# cd $SLURM_JOBTMP


python pipeline_dream10.py
