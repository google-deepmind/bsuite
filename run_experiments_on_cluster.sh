#!/bin/bash
#SBATCH -p bosch_cpu-cascadelake # ml_cpu-ivy # partition (queue)
#SBATCH -t 0-01:00 # time (D-HH:MM)
#SBATCH -c 1 # number of CPUs/task
#SBATCH -o log/%x.%A.%a.out # STDOUT  (the folder log has to exist!)  %N replaced by node name, %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -e log/%x.%A.%a.err # STDERR  (the folder log has to exist!)  %A will be replaced by the SLURM_ARRAY_JOB_ID value, whilst %a will be replaced by the SLURM_ARRAY_TASK_ID
#SBATCH -J bsuite-mdpp-merge-job-array # sets the job name. If not specified, the file name will be used as job name
#SBATCH -D /home/rajanr/bsuite_mdpp # Change working_dir (I think this directory _has_ to exist and won't be created!)
##SBATCH --mail-type=END,FAIL # (receive mails about end and timeouts/crashes of your job)
##SBATCH --gres=gpu:1  # reserves one GPU
##SBATCH --mem 16000M # Specify the real memory required per node, not needed as for our cluster, -c below takes priority and auto-sets the memory. For CPU, use --mem-per-cpu
#SBATCH -a 0-19 # Sets SLURM_ARRAY_TASK_ID - array index values, e.g. 0-31:2; 0-11%4 (it means max 4 tasks at a time)

export EXP_NAME='mdp_playground_p_noise' # Ideally contains Area of research + algorithm + dataset # Could just pass this as job name?
export AGENT='tf/boot_dqn'

echo -e '\033[32m'
# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with Job ID: $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";
echo "SLURM_CONF location: ${SLURM_CONF}"
echo "SLURM_JOB_NODELIST = ${SLURM_JOB_NODELIST}"
#cat /proc/cpuinfo
#cat /proc/meminfo
#df -h
/bin/hostname -f
#srun -n3 /bin/hostname -f # srun command not found on cluster!

python3 -V

#export PATH="/home/rajanr/anaconda2/bin:$PATH"
echo Paths: $PATH
echo Parent program $0
echo Shell used is $SHELL
# type -a source

# source activate /home/rajanr/anaconda2/envs/py36
# source activate /home/rajanr/anaconda3/envs/py36_toy_rl
. /home/rajanr/anaconda3/etc/profile.d/conda.sh # for anaconda3
conda activate /home/rajanr/anaconda3/envs/bsuite_mdpp # should be conda activate and not source when using anaconda3?
#/home/rajanr/anaconda3/bin/conda activate /home/rajanr/anaconda2/envs/py36
which python
python -V
which python3
python3 -V
ping google.com -c 3

echo -e '\033[0m'
echo -e "Script file start:\n====================="
cat $0
echo -e "\n======================\nScript file end!"

# TODO
echo "Line common to all tasks with SLURM_JOB_ID: ${SLURM_JOB_ID}, SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}, SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

# ================================================== #
# Begin actual Code
mkdir -p bsuite_mdpp_${SLURM_ARRAY_JOB_ID}
cd bsuite_mdpp_${SLURM_ARRAY_JOB_ID}
# cd /home/rajanr/mdpp
\time -v python3 /home/rajanr/git/bsuite-mdpp-merge/bsuite/baselines/${AGENT}/run.py --bsuite_id=${EXP_NAME}/${SLURM_ARRAY_TASK_ID} --logging_mode=csv --verbose=False --save_path=.

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";
