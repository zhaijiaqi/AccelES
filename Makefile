#######################################################################################################################################
#
#	Basic Makefile for Vitis 2023.2
#	Usage make [emulation | build | clean | clean_sw_emu | clean_hw_emu | clean_hw | cleanall] TARGET=<sw_emu | hw_emu | hw>
#
#
#######################################################################################################################################

XOCC=v++
CC=g++

#############################
# Define files to compile ###
#############################

# Host code
HOST_SRC=./src/fpga/src/host_spmv_ucsr.cpp
HOST_HEADER_DIRS=.

# Host header files (optional, used to check if rebuild is required);
UTILS_DIR=./src/common/utils
FPGA_DIR=./src/fpga/src
HOST_HEADERS=./src/common/csc_matrix/csc_matrix.hpp $(UTILS_DIR)/evaluation_utils.hpp $(UTILS_DIR)/mmio.hpp $(UTILS_DIR)/options.hpp $(UTILS_DIR)/utils.hpp $(FPGA_DIR)/aligned_allocator.h $(FPGA_DIR)/gold_algorithms/gold_algorithms.hpp $(FPGA_DIR)/opencl_utils.hpp $(FPGA_DIR)/ip/coo_fpga.hpp

# config file
# CONFIG_FILE = ./myconfig.cfg

# Name of host executable
HOST_EXE=spmv_coo_hbm_topk_multicore_mega_main

# Kernel
KERNEL_DIR=./src/fpga/src/ip/spmv
KERNEL_SRC=$(KERNEL_DIR)/spmv_ucsr_top_k_multicore.cpp 
KERNEL_HEADER_DIRS=./src/fpga/src/ip/spmv
KERNEL_FLAGS=
# Name of the xclbin;
KERNEL_EXE=spmv_ucsr_top_k_main
# Name of the main kernel function to build;
KERNEL_NAME=spmv_ucsr_top_k_main

#############################
# Define FPGA & host flags  #
#############################

# Target clock of the FPGA, in MHz;
TARGET_CLOCK=225
# Port width, in bit, of the kernel;
PORT_WIDTH=512

# Device code for Alveo U200;
# ALVEO_U280=xilinx_u280_xdma_201920_3
# ALVEO_U280_DEVICE="\"xilinx_u280_xdma_201920_3"\"
ALVEO_U280=xilinx_u280_gen3x16_xdma_1_202211_1
ALVEO_U280_DEVICE="\"xilinx_u280_gen3x16_xdma_1_202211_1"\"
TARGET_DEVICE=$(ALVEO_U280)

# Flags to provide to xocc, specify here associations between memory bundles and physical memory banks.
# Documentation: https://www.xilinx.com/html_docs/xilinx2019_1/sdaccel_doc/wrj1504034328013.html
KERNEL_LDCLFLAGS=--xp param:compiler.preserveHlsOutput=1 \
	--connectivity.nk $(KERNEL_NAME):32:$(KERNEL_NAME)_1.$(KERNEL_NAME)_2.$(KERNEL_NAME)_3.$(KERNEL_NAME)_4.$(KERNEL_NAME)_5.$(KERNEL_NAME)_6.$(KERNEL_NAME)_7.$(KERNEL_NAME)_8.$(KERNEL_NAME)_9.$(KERNEL_NAME)_10.$(KERNEL_NAME)_11.$(KERNEL_NAME)_12.$(KERNEL_NAME)_13.$(KERNEL_NAME)_14.$(KERNEL_NAME)_15.$(KERNEL_NAME)_16.$(KERNEL_NAME)_17.$(KERNEL_NAME)_18.$(KERNEL_NAME)_19.$(KERNEL_NAME)_20.$(KERNEL_NAME)_21.$(KERNEL_NAME)_22.$(KERNEL_NAME)_23.$(KERNEL_NAME)_24.$(KERNEL_NAME)_25.$(KERNEL_NAME)_26.$(KERNEL_NAME)_27.$(KERNEL_NAME)_28.$(KERNEL_NAME)_29.$(KERNEL_NAME)_30.$(KERNEL_NAME)_31.$(KERNEL_NAME)_32\
	--connectivity.slr $(KERNEL_NAME)_1:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_2:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_3:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_4:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_5:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_6:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_7:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_8:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_9:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_10:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_11:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_12:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_13:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_14:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_15:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_16:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_17:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_18:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_19:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_20:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_21:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_22:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_23:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_24:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_25:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_26:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_27:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_28:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_29:SLR1 \
	--connectivity.slr $(KERNEL_NAME)_30:SLR2 \
	--connectivity.slr $(KERNEL_NAME)_31:SLR0 \
	--connectivity.slr $(KERNEL_NAME)_32:SLR1 \
	--connectivity.sp $(KERNEL_NAME)_1.m_axi_gmem0:HBM[0] \
	--connectivity.sp $(KERNEL_NAME)_2.m_axi_gmem0:HBM[1] \
	--connectivity.sp $(KERNEL_NAME)_3.m_axi_gmem0:HBM[2] \
	--connectivity.sp $(KERNEL_NAME)_4.m_axi_gmem0:HBM[3] \
	--connectivity.sp $(KERNEL_NAME)_5.m_axi_gmem0:HBM[4] \
	--connectivity.sp $(KERNEL_NAME)_6.m_axi_gmem0:HBM[5] \
	--connectivity.sp $(KERNEL_NAME)_7.m_axi_gmem0:HBM[6] \
	--connectivity.sp $(KERNEL_NAME)_8.m_axi_gmem0:HBM[7] \
	--connectivity.sp $(KERNEL_NAME)_9.m_axi_gmem0:HBM[8] \
 	--connectivity.sp $(KERNEL_NAME)_10.m_axi_gmem0:HBM[9] \
	--connectivity.sp $(KERNEL_NAME)_11.m_axi_gmem0:HBM[10] \
	--connectivity.sp $(KERNEL_NAME)_12.m_axi_gmem0:HBM[11] \
	--connectivity.sp $(KERNEL_NAME)_13.m_axi_gmem0:HBM[12] \
	--connectivity.sp $(KERNEL_NAME)_14.m_axi_gmem0:HBM[13] \
	--connectivity.sp $(KERNEL_NAME)_15.m_axi_gmem0:HBM[14] \
	--connectivity.sp $(KERNEL_NAME)_16.m_axi_gmem0:HBM[15] \
	--connectivity.sp $(KERNEL_NAME)_17.m_axi_gmem0:HBM[16] \
	--connectivity.sp $(KERNEL_NAME)_18.m_axi_gmem0:HBM[17] \
	--connectivity.sp $(KERNEL_NAME)_19.m_axi_gmem0:HBM[18] \
	--connectivity.sp $(KERNEL_NAME)_20.m_axi_gmem0:HBM[19] \
	--connectivity.sp $(KERNEL_NAME)_21.m_axi_gmem0:HBM[20] \
 	--connectivity.sp $(KERNEL_NAME)_22.m_axi_gmem0:HBM[21] \
	--connectivity.sp $(KERNEL_NAME)_23.m_axi_gmem0:HBM[22] \
	--connectivity.sp $(KERNEL_NAME)_24.m_axi_gmem0:HBM[23] \
	--connectivity.sp $(KERNEL_NAME)_25.m_axi_gmem0:HBM[24] \
	--connectivity.sp $(KERNEL_NAME)_26.m_axi_gmem0:HBM[25] \
	--connectivity.sp $(KERNEL_NAME)_27.m_axi_gmem0:HBM[26] \
	--connectivity.sp $(KERNEL_NAME)_28.m_axi_gmem0:HBM[27] \
	--connectivity.sp $(KERNEL_NAME)_29.m_axi_gmem0:HBM[28] \
 	--connectivity.sp $(KERNEL_NAME)_30.m_axi_gmem0:HBM[29] \
	--connectivity.sp $(KERNEL_NAME)_31.m_axi_gmem0:HBM[30] \
	--connectivity.sp $(KERNEL_NAME)_32.m_axi_gmem0:HBM[31] \

KERNEL_ADDITIONAL_FLAGS=--kernel_frequency $(TARGET_CLOCK) -j 40 -O3

#  Specify host compile flags and linker;
HOST_INCLUDES= -I${XILINX_XRT}/include -I${XILINX_VITIS}/include
HOST_CFLAGS=$(HOST_INCLUDES) -D TARGET_DEVICE=$(ALVEO_U280_DEVICE) -g -D C_KERNEL -O3 -std=c++1y -pthread -lrt -lstdc++ 
HOST_LFLAGS=-L${XILINX_XRT}/lib -lxilinxopencl -lOpenCL

##########################################
# No need to modify starting from here ###
##########################################

#############################
# Define compilation type ###
#############################

# TARGET for compilation [sw_emu | hw_emu | hw | host]
TARGET=none
REPORT_FLAG=n
REPORT=
ifeq (${TARGET}, sw_emu)
$(info software emulation)
TARGET=sw_emu
ifeq (${REPORT_FLAG}, y)
$(info creating REPORT for software emulation set to true. This is going to take longer as it will synthesize the kernel)
REPORT=--report estimate
else
$(info I am not creating a REPORT for software emulation, set REPORT_FLAG=y if you want it)
REPORT=
endif
else ifeq (${TARGET}, hw_emu)
$(info hardware emulation)
TARGET=hw_emu
REPORT=--report estimate
else ifeq (${TARGET}, hw)
$(info system build)
TARGET=hw
REPORT=--report system
else
$(info no TARGET selected)
endif

PERIOD:= :
UNDERSCORE:= _
DEST_DIR=build/$(TARGET)/$(subst $(PERIOD),$(UNDERSCORE),$(TARGET_DEVICE))

#############################
# Define targets ############
#############################

clean:
	rm -rf .Xil emconfig.json 

clean_sw_emu: clean
	rm -rf sw_emu
clean_hw_emu: clean
	rm -rf hw_emu
clean_hw: clean
	rm -rf hw

cleanall: clean_sw_emu clean_hw_emu clean_hw
	rm -rf _xocc_* xcl_design_wrapper_*

check_TARGET:
ifeq (${TARGET}, none)
	$(error Target can not be set to none)
endif

host:  check_TARGET $(HOST_SRC) $(HOST_HEADERS)
	mkdir -p $(DEST_DIR)
	$(CC) -g $(HOST_SRC) $(HOST_CFLAGS) $(HOST_LFLAGS) -o $(DEST_DIR)/$(HOST_EXE)

xo:	check_TARGET
	mkdir -p $(DEST_DIR)
	$(XOCC) -g --platform $(TARGET_DEVICE) --target $(TARGET) --compile --include $(KERNEL_HEADER_DIRS) --save-temps $(REPORT) --kernel $(KERNEL_NAME) $(KERNEL_SRC) $(KERNEL_LDCLFLAGS) $(KERNEL_FLAGS) $(KERNEL_ADDITIONAL_FLAGS) --output $(DEST_DIR)/$(KERNEL_EXE).xo 

xclbin:  check_TARGET xo
	$(XOCC) -g --platform $(TARGET_DEVICE) --target $(TARGET) --link --include $(KERNEL_HEADER_DIRS) --save-temps $(REPORT) --kernel $(KERNEL_NAME) $(DEST_DIR)/$(KERNEL_EXE).xo $(KERNEL_LDCLFLAGS) $(KERNEL_FLAGS) $(KERNEL_ADDITIONAL_FLAGS) --output $(DEST_DIR)/$(KERNEL_EXE).xclbin

emulation:  host xclbin
	emconfigutil --platform $(TARGET_DEVICE) --nd 1
	mv emconfig.json $(DEST_DIR)
	export XCL_EMULATION_MODE=$(TARGET)
	./$(DEST_DIR)/$(HOST_EXE) -x $(DEST_DIR)/$(KERNEL_EXE).xclbin
	$(info Remeber to export XCL_EMULATION_MODE=$(TARGET) and run emconfigutil for emulation purposes)

build:  host xclbin

run_system:  build
	./$(DEST_DIR)/$(HOST_EXE) -x $(DEST_DIR)/$(KERNEL_EXE).xclbin
