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

# host compiler
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda

# Locate nvcc: prefer $(CUDA_PATH)/bin/nvcc if executable, otherwise fall back to
# the nvcc found in PATH (if any). This avoids hard-failing on compute nodes
# where CUDA is provided via modules or a different path.
NVCC_CANDIDATE := $(CUDA_PATH)/bin/nvcc
ifeq ($(shell test -x $(NVCC_CANDIDATE) && echo yes),yes)
	NVCC := $(NVCC_CANDIDATE) -ccbin $(HOST_COMPILER)
else
	NVCC_PATH := $(shell command -v nvcc 2>/dev/null)
	ifneq ($(NVCC_PATH),)
		NVCC := $(NVCC_PATH) -ccbin $(HOST_COMPILER)
	else
		# Fallback to CUDA_PATH even if nvcc missing; make will then error with a
		# clearer message when trying to execute NVCC. Users should set CUDA_PATH
		# or load the appropriate module on compute nodes.
		NVCC := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
	endif
endif

# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     :=
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
# Common includes and paths for CUDA
INCLUDES  := -I/usr/local/cuda/include/ 
# By default do not force-link CUDA libraries here. Leave LIBRARIES empty so
# users who don't want cuBLAS/cuSPARSE can build without them. If you need to
# link cuBLAS/cuSPARSE or other libraries, pass LIBRARIES on the make command
# line, e.g.:
#
#   make LIBRARIES="-L/path/to/spmm/build -lSpMM_API -L/your/cuda/lib64 -lcublas -lcusparse"
#
LIBRARIES :=
# #
# ifneq ($(SPUTNIK_PATH),)
# 	INCLUDES  += -I${SPUTNIK_PATH}
# 	LIBRARIES += -L${SPUTNIK_PATH}/build/sputnik/ -lsputnik -lglog 
# endif
#
ifneq ($(SPMM_PATH),)
INCLUDES  += -I${SPMM_PATH}/build
LIBRARIES += -L${SPMM_PATH}/build -lSpMM_API
endif

################################################################################

# Gencode arguments
SMS ?= 80
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += --threads 0 --std=c++11
ALL_CCFLAGS += -maxrregcount=255 
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################
# Target rules
all: spmm_test

spmm_test.o:  spmm_test.cu spmm_test_utils.h sparTA.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

spmm_test: spmm_test.o 
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) 

clean:
	rm -f spmm_test spmm_test.o

.PHONY: check-env
check-env:
	@echo "NVCC: $(NVCC)"
	@echo "CUDA_PATH: $(CUDA_PATH)"
	@echo "HOST_COMPILER: $(HOST_COMPILER)"
	@echo "INCLUDES: $(INCLUDES)"
