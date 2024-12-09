#pragma once

#include <iostream>
#include "hls_stream.h"
#include "spmv_utils.hpp"
#include "../fpga_utils.hpp"
#include "../fpga_types.hpp"

/////////////////////////////
/////////////////////////////

// Used to handle partial results of each COO packet;
typedef struct {
	q_type_inout aggregated_res[Q_LIMITED_FINISHED_ROWS];
	int_type num_rows_in_packet;
	bool_type fin;
} reduction_result_topk;

// HLS cannot correctly partition arrays in structs, so we have to resort to "ap_uint" for some of the streams;
#define INT_SIZE 32
typedef ap_uint<6> finished_rows_block;
typedef ap_uint<ULTRACSR_PACKET_SIZE * Q_AP_INT_ROW_BITWIDTH> x_block;
typedef ap_uint<(Q_LIMITED_FINISHED_ROWS + 1) * Q_FIXED_WIDTH> aggregated_res_block;

/////////////////////////////
/////////////////////////////

// 插入假操作增加模块延时（pipeline深度）
template<typename T>
T reg(T d) {
#pragma HLS pipeline II=1
#pragma HLS inline off
#pragma HLS latency max=1 min=1 // 防止函数被合并优化
    return d;
}

#define MIN(res, a, b) (((res[(a)]) < (res[(b)])) ? (a) : (b))

void argmin_1(q_type_inout *res, x_bscsr *worst_idx, q_type_inout *worst_res) {
#pragma HLS pipeline II=1
	*worst_idx = 0;
	*worst_res = res[0];
}

void argmin_2(q_type_inout *res, x_bscsr *worst_idx, q_type_inout *worst_res) {
#pragma HLS pipeline II=1
	*worst_idx = MIN(res, 0, 1);
	*worst_res = res[*worst_idx];
}

void argmin_4(q_type_inout *res, x_bscsr *worst_idx, q_type_inout *worst_res) {
#pragma HLS pipeline II=1
	x_bscsr m0 = MIN(res, 0, 1);
	x_bscsr m1 = MIN(res, 2, 3);

	*worst_idx = MIN(res, m0, m1);
	*worst_res = res[*worst_idx];
}

inline void argmin_8(q_type_inout* res, x_bscsr *worst_idx, q_type_inout *worst_res) {
#pragma HLS inline
    x_bscsr m0 = MIN(res, 0, 1);
	x_bscsr m1 = MIN(res, 2, 3);
	x_bscsr m2 = MIN(res, 4, 5);
    x_bscsr m3 = MIN(res, 6, 7);

	x_bscsr m01 = MIN(res, m0, m1);
    x_bscsr m23 = MIN(res, m2, m3);

    *worst_idx = MIN(res, m01, m23);
    *worst_res = res[*worst_idx];
}

void argmin_16(q_type_inout *res, x_bscsr *worst_idx, q_type_inout *worst_res) {
#pragma HLS pipeline II=1
    x_bscsr m0 = MIN(res, 0, 1);
	x_bscsr m1 = MIN(res, 2, 3);
	x_bscsr m2 = MIN(res, 4, 5);
	x_bscsr m3 = MIN(res, 6, 7);

	x_bscsr m4 = MIN(res, 8, 9);
	x_bscsr m5 = MIN(res, 10, 11);
	x_bscsr m6 = MIN(res, 12, 13);
	x_bscsr m7 = MIN(res, 14, 15);

	x_bscsr m01 = MIN(res, m0, m1);
	x_bscsr m23 = MIN(res, m2, m3);
	x_bscsr m45 = MIN(res, m4, m5);
	x_bscsr m67 = MIN(res, m6, m7);

	x_bscsr m0123 = MIN(res, m01, m23);
	x_bscsr m4567 = MIN(res, m45, m67);

	*worst_idx = MIN(res, m0123, m4567);
	*worst_res = res[*worst_idx];
}

template <uint k>
void argmin(q_type_inout *res, x_bscsr *worst_idx, q_type_inout *worst_res) {
#pragma HLS pipeline II=1
	int_type curr_min = 0;
	for (uint i = 0; i < k; i++) {
#pragma HLS unroll
		curr_min = MIN(res, curr_min, i);
	}
	*worst_idx = curr_min;
	*worst_res = res[curr_min];
}

////////////////////////////////////////
// 在FPGA上执行稀疏矩阵向量乘积操作
////////////////////////////////////////
inline void inner_spmv_topk_product_stream(
		hls::stream<input_packet_int_x_ultracsr> &x,
		hls::stream<input_packet_qtype_ultracsr> &val,
		hls::stream<input_packet_qtype_ultracsr> &vec,
		hls::stream<reduction_result_topk> &aggregated_res, // 存放一个数据包的spmv结果
		hls::stream<bool_type> &xf,
		hls::stream<input_packet_int_x_ultracsr> &x_out) {

    input_packet_int_x_ultracsr x_local;
#pragma HLS RESOURCE variable=x_local core=RAM_2P_BRAM latency=1
#pragma HLS array_partition variable=x_local complete dim=1
    input_packet_qtype_ultracsr val_local;
#pragma HLS RESOURCE variable=val_local core=RAM_2P_BRAM latency=1
#pragma HLS array_partition variable=val_local complete dim=1
    input_packet_qtype_ultracsr vec_local;
#pragma HLS RESOURCE variable=vec_local core=RAM_2P_BRAM latency=1
#pragma HLS array_partition variable=vec_local complete dim=1
    q_type_inout pointwise_res_local[ULTRACSR_PACKET_SIZE];
#pragma HLS RESOURCE variable=pointwise_res_local core=RAM_2P_BRAM
#pragma HLS array_partition variable=pointwise_res_local complete dim=1

    x_local = x.read();
    val_local = val.read();
    vec_local = vec.read();

	// Point-wise multiplication of a chunk of "val" and "scattered_vec"; 
    // 计算该数据包的nnz值和对应的向量的点积结果
    q_type val_float, vec_float;
	POINTWISE: for (x_out_ultracsr k = 0; k < ULTRACSR_PACKET_SIZE; k++) {
#pragma HLS unroll
		val_float = val_local[k];
	    vec_float = vec_local[k];
        pointwise_res_local[k] = val_float * vec_float;
    }

	reduction_result_topk result;   // 保存同一行的spmv结果
#pragma HLS RESOURCE variable=result core=RAM_2P_BRAM
#pragma HLS array_partition variable=result complete dim=1

    // 聚合同一行元素的spmv结果
    int_type num_rows_in_packet = 1;
    x_out_ultracsr first = x_local[0];
    q_type_inout aggregator = 0;

// FF:12 LUT:12 URAM:7 BRAM:5
    x_out_ultracsr pre = first, cur = first; 
    for (x_out_ultracsr i = 0; i < ULTRACSR_PACKET_SIZE; i++) {
#pragma HLS unroll
        pre = (i > 0) ? x_local[i - 1] : first;
        cur = x_local[i]; 
        num_rows_in_packet += (cur - pre);
        aggregator = (pre == cur) ? aggregator + pointwise_res_local[i] : 0;
        result.aggregated_res[cur-first] = aggregator;
    }

    result.num_rows_in_packet = num_rows_in_packet;
	result.fin = xf.read();

	x_out << x_local;
	aggregated_res << result;
}

/////////////////////////////
// 重置结果缓冲区
/////////////////////////////
// double res buffer for minikernel and ultrakernel
// minikernel_buffer
inline void reset_buffer(q_type_inout res[Q_LIMITED_FINISHED_ROWS][K], int_type res_idx[LIMITED_FINISHED_ROWS][K]) {
	WRITE_LOCAL_1: for (x_out_ultracsr j = 0; j < Q_LIMITED_FINISHED_ROWS; j++) {
#pragma HLS unroll
		WRITE_LOCAL_2: for (x_out_ultracsr k = 0; k < K; k++) {
#pragma HLS unroll
			res[j][k] = (q_type_inout) 0.0;
			res_idx[j][k] = 0;
		}
	}
}

/////////////////////////////////////////////////////////////////
// LOOP_1解压缩数据包，将值存在片上的x_local，y_local，val_local中
// 然后将值散布到val_stream，x_stream，vec_stream和x_f_stream流中
/////////////////////////////////////////////////////////////////
inline void spmv_coo_loop_1(
        int_type num_packets_coo,
        q_input_block *coo,
        hls::stream<input_packet_int_x_ultracsr> &x_stream,
        hls::stream<input_packet_qtype_ultracsr> &vec_stream,
        hls::stream<input_packet_qtype_ultracsr> &val_stream,
        hls::stream<bool_type> &x_f_stream, // 记录是否是新的一行（原文中的new_row标记位）
        q_type vec[ULTRACSR_PACKET_SIZE][MAX_COLS]
        ) {
        // std::cout << "start loop 1 " << std::endl;
#pragma HLS inline
    x_out_ultracsr x_local[ULTRACSR_PACKET_SIZE] = {0};
#pragma HLS array_partition variable=x_local complete   // 将数组完全打散，以寄存器的方式实现，可同时获取所有数据
//#pragma HLS stable variable=x_local
    static y_ultracsr y_local[ULTRACSR_PACKET_SIZE];
#pragma HLS array_partition variable=y_local complete
//#pragma HLS stable variable=y_local
    static q_type_inout val_local[ULTRACSR_PACKET_SIZE];
#pragma HLS array_partition variable=val_local complete
//#pragma HLS stable variable=val_local

LOOP_1: for (int_type i = 0; i < num_packets_coo; i++) {
    // num_packets_coo 未知，因此使用loop_tripcount指导工具分析循环时延，帮助为设计提供合适的优化措施
#pragma HLS loop_tripcount min=hls_iterations_nnz max=hls_iterations_nnz avg=hls_iterations_nnz 
#pragma HLS pipeline II=hls_pipeline
        input_packet_int_x_ultracsr x_local_out;
        input_packet_int_y_ultracsr y_local_out;
        input_packet_qtype_ultracsr val_local_out;
        input_packet_qtype_ultracsr vec_local;  // 将参与计算的稠密vec中的某些元素取出来，放到vec_local中，便于后面计算

        bool_type x_f_local;
        // Read chunks of "x", "y", "val", then scatter values of "vec";
        q_input_block coo_local = coo[i];
        Q_read_block_x(coo_local, x_local); // 2~5 clc
        Q_read_block_y(coo_local, y_local); // 1 clc 
        Q_read_block_val(coo_local, val_local); // 1 clc
        Q_read_block_xf(coo_local, &x_f_local); // 1 clc
        // Q_read_block_pack(coo_local);
        READ_COO: for (x_out_ultracsr j = 0; j < ULTRACSR_PACKET_SIZE; j++) {
#pragma HLS unroll
            
            x_local_out[j] = x_local[j];
            y_local_out[j] = y_local[j];
            val_local_out[j] = val_local[j];
            vec_local[j] = vec[j / 2][y_local[j]];  // 将参与计算的稠密vec中的某些元素取出来，放到vec_local中，便于后面计算
        }
        // 并数据传递到下一阶段的处理流中
        x_stream << x_local_out;

        val_stream << val_local_out;
        vec_stream << vec_local;
        x_f_stream << x_f_local;
}
    
    // std::cout << "finish loop 1 " << std::endl;
}

//////////////////////////////////////////////////////////
// 按数据包执行稀疏矩阵向量乘积操作，并聚合相同行的点乘结果
//////////////////////////////////////////////////////////
inline void spmv_coo_loop_2(
        int_type num_packets_coo,
        hls::stream<input_packet_int_x_ultracsr> &x_stream,
        hls::stream<input_packet_qtype_ultracsr> &vec_stream,
        hls::stream<input_packet_qtype_ultracsr> &val_stream,
        hls::stream<reduction_result_topk> &aggregated_res_stream,
        hls::stream<bool_type> &x_f_stream,
        hls::stream<input_packet_int_x_ultracsr> &x_stream_out
        ) {
    // std::cout << "start loop2" << std::endl;
#pragma HLS inline
    LOOP_2: for (int_type i = 0; i < num_packets_coo; i++) {
#pragma HLS loop_tripcount min=hls_iterations_nnz max=hls_iterations_nnz avg=hls_iterations_nnz
#pragma HLS pipeline II=hls_pipeline
        // Perform point-wise products;
        inner_spmv_topk_product_stream(x_stream, val_stream, vec_stream, aggregated_res_stream, x_f_stream, x_stream_out);
    }
}

//////////////////////////////////////////////////////////
// 结合上一个数据包的最后一行元素是否完成聚合，来完成本次数据包的聚合
//////////////////////////////////////////////////////////
inline void spmv_coo_loop_3(
        int_type num_packets_coo,
        hls::stream<reduction_result_topk> &aggregated_res_stream,
        hls::stream<input_packet_int_x_ultracsr> &x_stream_out,
        hls::stream<finished_rows_block> &finished_rows_stream,
        hls::stream<int_type> &start_x_stream,
        hls::stream<aggregated_res_block> &aggregated_res_local_stream,
        q_type_inout aggregated_res_local[ULTRACSR_PACKET_SIZE]
        ) {
    // std::cout << "start loop 3" << std::endl;
#pragma HLS inline  // 内联优化，将函数代码插入到函数调用的地方，而不是通过栈来函数调用，从而减少函数调用开销
    // Support array used in the storage FSM.
    // Values written in it are the same for each column,
    //   but we need an array to have independent parallel R/W access;
    int_type last_row_of_packet = 0;
    q_type_inout last_row_of_packet_output = (q_type_inout) 0;
    finished_rows_block finished_rows = 0;

    LOOP_3: for (int_type i = 0; i < num_packets_coo; i++) {
#pragma HLS loop_tripcount min=hls_iterations_nnz max=hls_iterations_nnz avg=hls_iterations_nnz
#pragma HLS pipeline II=hls_pipeline
        // Reset local storage buffers;
        RESET_BUFFERS: for (x_out_ultracsr j = 0; j < Q_LIMITED_FINISHED_ROWS + 1; j++) {
#pragma HLS unroll
            aggregated_res_local[j] = 0;
            finished_rows = 0;
        }
        // Read the aggregated stream output;
        reduction_result_topk reduction_result_local = aggregated_res_stream.read();    // 读取流进来的loop2的聚合结果
        input_packet_int_x_ultracsr x_packet_tmp = x_stream_out.read();    // 读取流进来的loop2解压的x坐标
        int_type packet_starts_with_new_row = (i != 0) ? (int_type) reduction_result_local.fin : 0; // 根据reduction_result_local.fin的值来确定数据包是否以新的一行开始
        int_type num_rows_in_packet = reduction_result_local.num_rows_in_packet;    // 读取数据包中实际有多少行元素
        int_type finished_rows_num = num_rows_in_packet + packet_starts_with_new_row - 1;   // 计算有多少行元素完成了聚合
        int_type start_row_of_packet = last_row_of_packet + packet_starts_with_new_row; // 计算数据包的起始行号
        last_row_of_packet += finished_rows_num; // 计算数据包的结束行号
        // std::cout << num_rows_in_packet << ' ' << finished_rows_num << ' ' << start_row_of_packet << ' ' << last_row_of_packet << std::endl;
        // Fused loop 缓存loop2的聚合结果并记录哪些行已经被聚合完了
        aggregated_res_local[1] = reduction_result_local.aggregated_res[0];  // Leave empty the first value;
        // All rows with x_packet_tmp != 0 are finished, except the last non-zero entry (position "num_rows_in_packet - 1" in x_packet_tmp);
        READ_AGG_AND_FINISHED_ROW_CHECK: for (x_out_ultracsr j = 1; j < Q_LIMITED_FINISHED_ROWS; j++) {
#pragma HLS unroll 
            aggregated_res_local[1+j] = reduction_result_local.aggregated_res[j]; 
        }

        FINISHED_ROW_CHECK: for (x_out_ultracsr j = 1; j < ULTRACSR_PACKET_SIZE; j++) {
#pragma HLS unroll 
            finished_rows += x_packet_tmp[j] != x_packet_tmp[j-1]; 
        }
        // aggregated_res_local[ULTRACSR_PACKET_SIZE] = reduction_result_local.aggregated_res[num_rows_in_packet];
        // If the last row in the previous packet was split between packets, update the first result in this packet;
        // 如果上一个数据包的最后一行是否被分割，说明这个数据包的第一行数据与上一个数据包需要聚合
        if (!packet_starts_with_new_row) {
            aggregated_res_local[1] += last_row_of_packet_output;   // 该数据包的第一行聚合结果要加上上一个数据包的最后一行的聚合结果
            aggregated_res_local[0] = 0;
        } else {    // 如果上一个数据包的最后一行已经完成聚合，在本次迭代中会存储其值并开始处理。
            aggregated_res_local[0] = last_row_of_packet_output;
            // finished_rows |= 1 << 5;
        }
        // Book-keeping at the end of processing a packet;
        last_row_of_packet_output = aggregated_res_local[num_rows_in_packet];   // 记录本次数据包的最后一行的聚合结果，用于下一个数据包的聚合
        // Prepare packets for the final stage of data-flow;
        aggregated_res_block aggregated_res_b;
        for (x_out_ultracsr j = 0; j < Q_LIMITED_FINISHED_ROWS + 1; j++) {
#pragma HLS unroll
            unsigned int lower_range = Q_FIXED_WIDTH * j;
            unsigned int upper_range = Q_FIXED_WIDTH * (j + 1) - 1;
            q_type_inout agg_tmp = aggregated_res_local[j];    // 读取本次数据包的聚合结果
            // std::cout << "agg_tmp: " << agg_tmp << ' ';
            aggregated_res_b.range(upper_range, lower_range) = *((q_type_inout *) &agg_tmp);    // 将聚合结果转换为bscsr数据类型
        }
        // std::cout << std::endl;
        // std::cout << "finish_rows: " << finished_rows_b << std::endl;
        // 将数据流出去到loop4阶段（哪些行完成了聚合、从哪一行开始聚合、聚合结果）
        finished_rows_stream << finished_rows;
        start_x_stream << start_row_of_packet;
        aggregated_res_local_stream << aggregated_res_b;
    }
    // std::cout << "finish loop 3 " << std::endl;
}

////////////////////////////////////////////////////////
// 对每个 COO 数据包进行处理，更新 Top-K 值。
////////////////////////////////////////////////////////
inline void spmv_coo_loop_4(
    int_type num_packets_coo,
    hls::stream<int_type>& start_x_stream,
    hls::stream<finished_rows_block>& finished_rows_stream,
    hls::stream<aggregated_res_block>& aggregated_res_local_stream,
    int_type res_idx[Q_LIMITED_FINISHED_ROWS][K],
    q_type_inout res[Q_LIMITED_FINISHED_ROWS][K],
    x_bscsr curr_worst_idx[Q_LIMITED_FINISHED_ROWS + 1],
    q_type_inout curr_worst_val[Q_LIMITED_FINISHED_ROWS + 1],
    q_type_inout aggregated_res_local_2[ULTRACSR_PACKET_SIZE],
    q_type_inout res_local[Q_LIMITED_FINISHED_ROWS + 1][K],
    int_type res_idx_local[Q_LIMITED_FINISHED_ROWS + 1][K],
    finished_rows_block finished_rows_b,
    hls::stream<x_bscsr> &worst_idx_stream,
    hls::stream<q_type_inout> &worst_val_stream
    ) {
#pragma HLS inline
    // finished_rows_block finished_rows_2;
    LOOP_4: for (int_type i = 0; i < num_packets_coo; i++) {
#pragma HLS loop_tripcount min=hls_iterations_nnz max=hls_iterations_nnz avg=hls_iterations_nnz
#pragma HLS pipeline II=hls_pipeline
        // 读取流入的数据流（从哪一行开始聚合、哪些行完成了聚合、聚合结果）
        int_type start_x = start_x_stream.read();
        finished_rows_b = finished_rows_stream.read();
        aggregated_res_block aggregated_res_b = aggregated_res_local_stream.read();    //  (fifo-read: 0.811 ns)   
        // Move values to local arrays;
        for (x_out_ultracsr j = 0; j < Q_LIMITED_FINISHED_ROWS + 1; j++) {
#pragma HLS unroll
            unsigned int lower_range = Q_FIXED_WIDTH * j;
            unsigned int upper_range = Q_FIXED_WIDTH * (j + 1) - 1;
            unsigned int val_curr = aggregated_res_b.range(upper_range, lower_range);
            aggregated_res_local_2[j] = *((q_type_inout*)&val_curr);
        }
        UPDATE_TOP_K: for (x_out_ultracsr j = 0; j < Q_LIMITED_FINISHED_ROWS; j++) {
#pragma HLS unroll
            q_type_inout curr_val = aggregated_res_local_2[j];
            if (curr_val >= curr_worst_val[j] && j <= finished_rows_b) {     // (icmp: 0.785 ns) (xor: 0.122ns) (and: 0.122ns)
                res_idx_local[j][curr_worst_idx[j]] = start_x + j - 1;
                res_local[j][curr_worst_idx[j]] = curr_val;
            }
            argmin_8(res_local[j], &curr_worst_idx[j], &curr_worst_val[j]);
        }

        // TODO sort K elements
        WRITE_LOCAL_2: for (x_out_ultracsr i = 0; i < Q_LIMITED_FINISHED_ROWS; i++) {
    #pragma HLS unroll
            WRITE_LOCAL_RES_3 : for (x_out_ultracsr j = 0; j < K; j++) {
    #pragma HLS unroll
                res[i][j] = res_local[i][j];
                res_idx[i][j] = res_idx_local[i][j];
            }
        }
    }
}



//////////////////////////////////////////////////////////
// 每个数据包进入芯片后的数据流loop1-4
//////////////////////////////////////////////////////////
inline void spmv_ucsr_top_k_multi_stream_inner(q_input_block *coo, int_type rows, int_type cols, int_type nnz,
		q_type vec[ULTRACSR_PACKET_SIZE][MAX_COLS], int_type res_idx[Q_LIMITED_FINISHED_ROWS][K], q_type_inout res[Q_LIMITED_FINISHED_ROWS][K], x_bscsr curr_worst_idx[Q_LIMITED_FINISHED_ROWS+1], q_type_inout curr_worst_val[LIMITED_FINISHED_ROWS+1]) {

#pragma HLS dataflow    // 使用数据流编程模型来设计函数中的计算流程，控制逻辑和数据处理逻辑是分离的，可以提高并行性和系统的吞吐量。
// 保持变量的值稳定，避免在时序逻辑中优化
#pragma HLS stable variable=res_idx
#pragma HLS stable variable=res
#pragma HLS stable variable=vec
#pragma HLS stable variable=curr_worst_idx
#pragma HLS stable variable=curr_worst_val

	int_type num_packets_coo = (nnz + ULTRACSR_PACKET_SIZE - 1) / ULTRACSR_PACKET_SIZE;   // 传输spm的数据包个数

    // Define streams; hls::stream<T> 表示AXI4‑Stream数据流，T表示数据类型，depth表示数据流的深度，即最大存储的数据包数量
    // AXI4-Stream没有与数据流相关的地址，它只是一个数据流，可以用于芯片内部的数据流传输，适用于高速大数据应用，比如视频数据流，相比较AXI4和AXI4-Lite，不限制突发长度。AXI主要面对内存映射
    hls::stream<input_packet_int_x_ultracsr> x_stream_1("x_stream_1");
#pragma HLS STREAM variable=x_stream_1 depth=1  // 创建数据流通道，深度为1，可以存储的最大数据包数量为1
    hls::stream<input_packet_qtype_ultracsr> val_stream_1("val_stream_1");
#pragma HLS STREAM variable=val_stream_1 depth=1
    hls::stream<input_packet_qtype_ultracsr> vec_stream_1("vec_stream_1");
#pragma HLS STREAM variable=vec_stream_1 depth=1

    hls::stream<input_packet_real_coo> pointwise_stream;
#pragma HLS STREAM variable=pointwise_stream depth=1
    hls::stream<reduction_result_topk> aggregated_res_stream;   // 用于记录loop2一个数据包的聚合结果
#pragma HLS STREAM variable=aggregated_res_stream depth=1

    hls::stream<bool_type> x_f_stream_1("x_f_stream_1");
#pragma HLS STREAM variable=x_f_stream_1 depth=1

    hls::stream<input_packet_int_x_ultracsr> x_stream_out;
#pragma HLS STREAM variable=x_stream_out depth=1

    hls::stream<finished_rows_block> finished_rows_stream;
#pragma HLS STREAM variable=finished_rows_stream depth=1
    hls::stream<aggregated_res_block> aggregated_res_local_stream;
#pragma HLS STREAM variable=aggregated_res_local_stream depth=1
    hls::stream<int_type> start_x_stream;
#pragma HLS STREAM variable=start_x_stream depth=1

    hls::stream<x_bscsr> worst_idx_stream;
#pragma HLS STREAM variable=worst_idx_stream depth=1
    hls::stream<q_type_inout> worst_val_stream;
#pragma HLS STREAM variable=worst_val_stream depth=1

    q_type_inout aggregated_res_local[Q_LIMITED_FINISHED_ROWS + 1];
#pragma HLS array_partition variable=aggregated_res_local complete dim=0
// 	bool finished_rows[Q_LIMITED_FINISHED_ROWS + 1];
// #pragma HLS array_partition variable=finished_rows complete dim=0

    q_type_inout aggregated_res_local_2[Q_LIMITED_FINISHED_ROWS + 1];
#pragma HLS array_partition variable=aggregated_res_local_2 complete dim=0
#pragma HLS stable variable=aggregated_res_local_2
    finished_rows_block finished_rows_b;
#pragma HLS array_partition variable=finished_rows_b complete dim=0
#pragma HLS stable variable=finished_rows_b

	// Store results;
	q_type_inout res_local[Q_LIMITED_FINISHED_ROWS+1][K];
#pragma HLS array_partition variable=res_local complete dim=0
//#pragma HLS stable variable=res_local

	int_type res_idx_local[Q_LIMITED_FINISHED_ROWS+1][K];
#pragma HLS array_partition variable=res_idx_local complete dim=0
//#pragma HLS stable variable=res_idx_local

	reset_buffer(res_local, res_idx_local);
    // 以下循环都以数据包为基本单位处理，循环次数为num_packets_coo
    // std::cout << "finish reset buffer" << std::endl;

    spmv_coo_loop_1(num_packets_coo, coo, x_stream_1, vec_stream_1, val_stream_1, x_f_stream_1, vec);
    // std::cout<< "loop1" <<std::endl;
    spmv_coo_loop_2(num_packets_coo, x_stream_1, vec_stream_1, val_stream_1, aggregated_res_stream, x_f_stream_1, x_stream_out);
    // std::cout<< "loop2" <<std::endl;
    spmv_coo_loop_3(num_packets_coo, aggregated_res_stream, x_stream_out, finished_rows_stream, start_x_stream, aggregated_res_local_stream, aggregated_res_local);
    // std::cout<< "loop3" <<std::endl;
    spmv_coo_loop_4(num_packets_coo,start_x_stream, finished_rows_stream, aggregated_res_local_stream, res_idx, res, curr_worst_idx, curr_worst_val, aggregated_res_local_2, res_local, res_idx_local, finished_rows_b, worst_idx_stream, worst_val_stream);
    // std::cout<< "loop4" <<std::endl;
}

/////////////////////////////
/////////////////////////////

inline void spmv_ucsr_top_k_multi_stream(q_input_block *coo, int_type rows, int_type cols, int_type nnz,
		q_type vec[ULTRACSR_PACKET_SIZE][MAX_COLS], int_type res_idx[Q_LIMITED_FINISHED_ROWS][K], q_type_inout res[Q_LIMITED_FINISHED_ROWS][K]) {
#pragma HLS inline

	// 跟踪top-k中最差的结果，用于loop4中替换当前最差结果
	x_bscsr curr_worst_idx[Q_LIMITED_FINISHED_ROWS+1];
#pragma HLS array_partition variable=curr_worst_idx complete dim=0

	q_type_inout curr_worst_val[Q_LIMITED_FINISHED_ROWS+1];
#pragma HLS array_partition variable=curr_worst_val complete dim=0

	for (x_out_ultracsr i = 0; i < Q_LIMITED_FINISHED_ROWS+1; i++) {
#pragma HLS unroll
		curr_worst_idx[i] = 0;
		curr_worst_val[i] = (q_type_inout) 0.0;
	}
    // std::cout << "finish top" << std::endl;

	// It's useful to wrap the data-flow function so that we can do some post-processing if required;
	spmv_ucsr_top_k_multi_stream_inner(coo, rows, cols, nnz, vec, res_idx, res, curr_worst_idx, curr_worst_val);
}

/////////////////////////////
/////////////////////////////

extern "C" void spmv_ucsr_top_k_main(
		q_input_block *coo0,
		int_type num_rows0,
		int_type num_cols0,
		int_type nnz0,
		vec_int_inout_ultracsr *vec,
		input_packet_int_ultracsr *res_idx0,
		input_packet_qtype_inout_ultracsr *res0
		);
