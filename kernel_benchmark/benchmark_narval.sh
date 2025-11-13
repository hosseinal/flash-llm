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

M=(21504  7168   28672  7168   27648  9216   36864  9216   36864  12288  49152  12288)
K=(7168   7168   7168   28672  9216   9216   9216   36864  12288  12288  12288  49152)
SplitK=(5      7      7      7      2      6      3      6      3      9      9     9)
N=(8 16 32 64 128)
Sparsity=(70 80 90 95)

rm -rf result_flashllm_large.txt

for BS in ${N[@]}
do
    #echo "BS=${BS}"
    for ((i=0;i<${#M[@]};i++))
    do
        #echo "Processing Shape ${i}..."
        for S in ${Sparsity[@]}
        do
            m=${M[i]}
            k=${K[i]}
            path="${BASEPATH}/matrix_${m}x${k}_sparsity_${S}.mtx"
            
            # ./spmm_test ${M[i]} ${K[i]} ${BS} ${S} ${SplitK[i]} >> result.txt

            ./spmm_test ${path} ${BS} ${SplitK[i]} >> result_flashllm_large.txt

        done
    done
done

