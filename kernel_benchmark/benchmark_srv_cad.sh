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
#SBATCH -t 10:00:00
#SBATCH --mem=50G  # Request 32 GB of memory

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


