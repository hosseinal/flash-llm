#!/bin/bash

# Copyright 2023 The FLash-LLM Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=1
#SBATCH --export=ALL
#SBATCH --job-name="compressed"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=hosseinalbakri3@gmail.com
#SBATCH --nodes=1
#SBATCH --output="compressed.%j.%N.out"
#SBATCH -t 96:00:00
#SBATCH --mem=50G  # Request 32 GB of memory

BASEPATH=$1
ListPath=$2

cd ..
source Init_FlashLLM.sh
cd kernel_benchmark
source test_env

module load StdEnv/2020
module load intel/2022.2.1
echo " +-+-+-+-+ ========> ${MKLROOT}"
export MKL_DIR=$MKLROOT
echo " +-+-+-+-+ ========> ${MKL_DIR}"
module load cmake
module load gcc
module load python
module load cuda/12.2


N=(8 16 32 64 128)
SplitK=(2 3 6 7 9)

rm -rf result_flashllm_dlmc.txt

for B in ${N[@]}
do
    for S in ${SplitK[@]}
    do
        while read path; do
         
            echo ${BASEPATH}${path}
            ./spmm_test ${BASEPATH}${path} ${B} ${S} >> result_flashllm_dlmc.txt

        done < ${ListPath}
    done
done



