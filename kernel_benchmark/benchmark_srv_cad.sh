#!/bin/bash

##################### SLURM (do not change) v  #####################
#SBATCH --export=ALL
#SBATCH --job-name="project"
#SBATCH --nodes=1
#SBATCH --output="project.%j.%N.out"
#SBATCH -t 01:00:00
##################### SLURM (do not change) ^  #####################

# Above are SLURM directives for job scheduling on a cluster,
export SLURM_CONF=/etc/slurm/slurm.conf

BASEPATH=$1

cd ..
source Init_FlashLLM.sh
cd kernel_benchmark
source test_env

module load StdEnv/202
module load cmake
module load gcc
module load python
module load cuda/12.2

BASEPATH=$1
MTX_FILE_PATH=$2

rm -rf result_flashllm_large.txt

N=(32 64 128)

# If an MTX file list is provided as the 2nd arg (MTX_FILE_PATH), use it.
# Each non-empty line in the list should contain the relative mtx path
# optionally followed by a SplitK integer. Lines starting with # are ignored.
# Example:
#   matrix_21504x7168_sparsity_70.mtx 5
#   matrix_7168x7168_sparsity_80.mtx
LIST_FILE=${MTX_FILE_PATH:-benchmark_stc_caf}
DEFAULT_SPLITK=1


while IFS= read -r raw || [ -n "${raw}" ]; do
    # strip comments and trim
    line="${raw%%#*}"
    line="$(echo -e "${line}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    [ -z "${line}" ] && continue

    # split into relpath [splitk]
    set -- ${line}
    relpath=$1
    splitk=${2:-$DEFAULT_SPLITK}

    for BS in ${N[@]}; do
        echo "Running: ${BASEPATH}/${relpath}  BS=${BS}  SplitK=${splitk}"
        ./spmm_test "${BASEPATH}/${relpath}" ${BS} ${splitk} >> result_flashllm_large.txt
    done
done < "${LIST_FILE}"


