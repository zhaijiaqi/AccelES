#include "spmv_ucsr_top_k_multicore.hpp"

/////////////////////////////
/////////////////////////////

#define USE_URAM_FOR_VEC true

void spmv_ucsr_top_k_main(
		q_input_block *coo0,
		int_type num_rows0,
		int_type num_cols0,
		int_type nnz0,
		vec_int_inout_ultracsr *vec,
		input_packet_int_ultracsr *res_idx0,	// 这里数据宽度为960，导致硬仿真接口为1024而非512
		input_packet_qtype_inout_ultracsr *res0
		) {
// 定义传输数据的协议，using AXI master;
#pragma HLS INTERFACE m_axi port = coo0 offset = slave bundle = gmem0 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = vec offset = slave bundle = gmem0 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = res_idx0 offset = slave bundle = gmem0 // num_write_outstanding = 32 latency = 100
#pragma HLS INTERFACE m_axi port = res0 offset = slave bundle = gmem0 // num_write_outstanding = 32 latency = 100

// #pragma HLS INTERFACE s_axilite register port = return bundle = control

// // Pragmas used for data-packing;
#pragma HLS data_pack variable=res_idx0 struct_level
#pragma HLS data_pack variable=res0 struct_level

	// std::cout << "start kernel" << std::endl;

////////////////////////////////////////////
//  声明片上资源（用于存放中间计算结果） /////////
////////////////////////////////////////////
	static q_type_inout res_local[SUB_SPMV_PARTITIONS][Q_LIMITED_FINISHED_ROWS][K];	// 静态数组buffer，用于存放计算结果，用寄存器存放
#pragma HLS array_partition variable=res_local complete dim=0

	static int_type res_local_idx[SUB_SPMV_PARTITIONS][Q_LIMITED_FINISHED_ROWS][K];	// 静态数组buffer，用于存放计算结果的索引，用寄存器存放
#pragma HLS array_partition variable=res_local_idx complete dim=0

	// Dense vector allocated in URAM;
	static q_type vec_local[SUB_SPMV_PARTITIONS][Q_VEC_REPLICAS][MAX_COLS];	// 用URAM存放的稠密向量V，每个super核中有4个小核，每个小核中复制了VEC_REPLICAS个向量，向量最大长度为1024
#pragma HLS RESOURCE variable=vec_local core=XPM_MEMORY uram					// 声明为URAM资源
#pragma HLS array_partition variable=vec_local complete dim=1
#pragma HLS array_partition variable=vec_local complete dim=2

	// Reset the local vector buffer;
	RESET_OUTPUT_0: for (int_type s = 0; s < SUB_SPMV_PARTITIONS; s++) {	// 并行地重置本地计算结果缓存
#pragma HLS unroll
		RESET_OUTPUT_1: for (int_type i = 0; i < Q_LIMITED_FINISHED_ROWS; i++) {
#pragma HLS unroll
			RESET_OUTPUT_2: for (int_type j = 0; j < K; j++) {
#pragma HLS unroll
				res_local[s][i][j] = 0.0;
				res_local_idx[s][i][j] = 0;
			}
		}
	}
	
	// std::cout << "read data" << std::endl;
////////////////////////////////////////////////////////
// 从HBM读入稠密向量V，每个小核内复制 VEC_REPLICAS 份 ///////
///////////////////////////////////////////////////////

	// 定义临时存储输入的vec，定义资源类型为URAM，512位宽的package存不下1024个数据，所以需要定义成一个数组，含义是最多需要94个数据包（vec长度为1024）才能把vec传到fpga的uranm上。
	static vec_int_inout_ultracsr vec_loader[(MAX_COLS + ULTRACSR_PACKET_SIZE - 1) / ULTRACSR_PACKET_SIZE];
#pragma HLS RESOURCE variable=vec_loader core=XPM_MEMORY uram
	int_type num_blocks_cols0 = (num_cols0 + ULTRACSR_PACKET_SIZE - 1) / ULTRACSR_PACKET_SIZE;	// 计算实际需要的数据包个数
	// Read the input "vec" and store it on a local buffer;
	// 采用流水线的方式从HBM读入Vec存放到片上uranm上
	// 每个循环周期读入一个ULTRACSR_PACKET_SIZE的向量数据，并存入本地缓存中。
	vec_int_inout_ultracsr vec_tmp;
	READ_INPUT: for (int_type i = 0; i < num_blocks_cols0; i++) {
    #pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=hls_iterations_cols max=hls_iterations_cols avg=hls_iterations_cols		// 用于设置循环迭代的预期次数范围(1024/8=128)。
		vec_loader[i] = vec[i];	// 读入数据包，vec_loader在fpga的uram上，vec在HBM内存中
		// std::cout << std::hex << std::uppercase;
		// std::cout << "vec_loader[" << i << "] in hexadecimal: " << vec_loader[i] << std::endl;
	}
	// Copy the input "vec" on all the other "vec" copies;
	// 将输入的vec复制到其他的vec副本中
COPY_INPUT: for (int_type i = 0; i < num_cols0; i++) {
#pragma HLS loop_tripcount min=hls_num_cols max=hls_num_cols avg=hls_num_cols	// 用1024个时钟将输入的vec复制到其他的vec副本中。
#pragma HLS pipeline II=1
		vec_int_inout_ultracsr vec_tmp = vec_loader[i / ULTRACSR_PACKET_SIZE];
		unsigned int lower_val_range = Q_AP_INT_VAL_BITWIDTH * (i % ULTRACSR_PACKET_SIZE);
		unsigned int upper_val_range = Q_AP_INT_VAL_BITWIDTH * ((i % ULTRACSR_PACKET_SIZE) + 1) - 1;
		unsigned int block_curr = vec_tmp.range(upper_val_range, lower_val_range);
		q_type curr = *((q_type*)&block_curr);
		// if(i>500)	std::cout << "curr[" << i << "] = " << curr << std::endl;

		for (int_type s = 0; s < SUB_SPMV_PARTITIONS; s++) {	// 大核中的4个小核都要并行地执行向量复制操作，每个小核中复制 VEC_REPLICAS 份
#pragma HLS unroll
			for (int_type j = 0; j < Q_VEC_REPLICAS; j++) {
#pragma HLS unroll
				vec_local[s][j][i] = (q_type) curr;
			}
		}
	}

/*
	std::cout << "vec_local[0][0]" << std::endl;
	for(int_type i=0; i<num_cols0; i++){
		std::cout << vec_local[0][0][i] << " ";
		if(i%ULTRACSR_PACKET_SIZE==0)	std::cout << std::endl;
	}
*/
	// std::cout << "finish read vec" << std::endl;

/////////////////////////////////
//  Main SpMV computation ///////
//  并行地执行4个小核的计算   ///////
/////////////////////////////////
	spmv_ucsr_top_k_multi_stream(coo0, num_rows0, num_cols0, nnz0, vec_local[0], res_local_idx[0], res_local[0]);
	
	// std::cout << "finish compute" << std::endl;
////////////////////////////////
////   将计算结果写回    /////////
///////////////////////////////
	
	WRITE_OUTPUT: for (int_type i = 0; i < K; i++) {
#pragma HLS pipeline II=hls_pipeline
		input_packet_int_ultracsr packet_idx0;
		input_packet_qtype_inout_ultracsr packet0;

// #pragma HLS bind_storage variable=packet_idx0 type=RAM_T2P impl=bram
// #pragma HLS bind_storage variable=packet_idx1 type=RAM_T2P impl=bram
// #pragma HLS bind_storage variable=packet_idx2 type=RAM_T2P impl=bram
// #pragma HLS bind_storage variable=packet_idx3 type=RAM_T2P impl=bram
// #pragma HLS bind_storage variable=packet0 type=RAM_T2P impl=bram
// #pragma HLS bind_storage variable=packet1 type=RAM_T2P impl=bram
// #pragma HLS bind_storage variable=packet2 type=RAM_T2P impl=bram
// #pragma HLS bind_storage variable=packet3 type=RAM_T2P impl=bram
		
		input_packet_qtype_inout_ultracsr packet;
		for (int_type j = 0; j < Q_LIMITED_FINISHED_ROWS; j++) {			// TODO:原来是 ULTRACSR_PACKET_SIZE
#pragma HLS unroll	
			packet_idx0[j] = res_local_idx[0][j][i];
			packet0[j] = (q_type_inout) res_local[0][j][i];
		}
		res_idx0[i] = packet_idx0;
		res0[i] = packet0;
	}
}