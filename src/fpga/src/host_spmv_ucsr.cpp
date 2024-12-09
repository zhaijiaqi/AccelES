#include <iostream>
#include <math.h>
#include <vector>
#include <getopt.h>
#include <chrono>
#include <random>
#include <CL/cl_ext.h>
#include <unordered_set>

#include "opencl_utils.hpp"
#include "../../common/utils/utils.hpp"
#include "../../common/utils/options.hpp"
#include "../../common/utils/evaluation_utils.hpp"
#include "../../common/utils/quantize_utils.hpp"
#include "../../common/types.hpp"
#include "ip/fpga_types.hpp"
#include "ip/coo_matrix.hpp"
#include "gold_algorithms/gold_algorithms.hpp"
#include "ip/fpga_utils.hpp"
#include "aligned_allocator.h"

/////////////////////////////
/////////////////////////////

namespace chrono = std::chrono;
using clock_type = chrono::high_resolution_clock;

/////////////////////////////
/////////////////////////////

// 表示超级稀疏矩阵向量乘法（SpMV）中的子分区。
struct SubSpMVPartition {
	int_type partition_id;	// 子SpMV分区的编号。
	std::vector<std::tuple<int_type, int_type, real_type_inout>> coo_partition;	// 子SpMV分区的COO数据。
	std::vector<input_block, aligned_allocator<input_block>> bscsr_in;	// 这里已经不是coo格式数据了，已经是打包好的BS-CSR格式数据了。
	std::vector<input_packet_int_bscsr, aligned_allocator<input_packet_int_bscsr>> res_idx_out;	// 输出结果的索引数据。
	std::vector<input_packet_real_inout_bscsr, aligned_allocator<input_packet_real_inout_bscsr>> res_out;	// 输出结果的实数数据。

	cl::Buffer res_idx_buf;	// 输出结果的索引数据的缓冲区。
	cl::Buffer res_buf;	// 输出结果的实数数据的缓冲区。

	// Each partition has a fixed number of rows and a number of nnz that depends on the rows.
	// We track the first and last rows associated with the partition to split the total COO nnz;
	int_type num_rows_partition;	// 子SpMV分区的行数。
	int_type num_nnz_partition;	// 子SpMV分区的非零元素个数。
	int_type first_row;	// 子SpMV分区的第一个行的编号。
	int_type last_row;	// 子SpMV分区的最后一行的编号。

	int_type num_blocks_rows;	// 子SpMV分区的行块数。
	int_type num_blocks_nnz;	// 传输子SpMV分区需要的数据包个数

	SubSpMVPartition(int_type _num_rows_partition, int_type _partition_id): num_rows_partition(_num_rows_partition), partition_id(_partition_id) {
		num_blocks_rows = (num_rows_partition + BSCSR_PACKET_SIZE - 1) / BSCSR_PACKET_SIZE;
		for (int_type i = 0; i < K * TOPK_RES_COPIES; i++) {
			input_packet_int_bscsr packet_res_idx;
			input_packet_real_inout_bscsr packet_res;
			res_idx_out.push_back(packet_res_idx);
			res_out.push_back(packet_res);
		}
	}
};

// 用于存放量化后SpM的子分区。 2024-05-22
struct Q_SubSpMVPartition {
	int_type partition_id;	// 子SpMV分区的编号。
	std::vector<std::tuple<int_type, int_type, q_type>> coo_partition;	// 子SpMV分区的COO数据。
	std::vector<q_input_block, aligned_allocator<q_input_block>> ultracsr_in;	// 这里已经不是coo格式数据了，已经是打包好的BS-CSR格式数据了。
	std::vector<input_packet_int_ultracsr, aligned_allocator<input_packet_int_ultracsr>> res_idx_out;	// 输出结果的索引数据。
	std::vector<input_packet_qtype_inout_ultracsr, aligned_allocator<input_packet_qtype_inout_ultracsr>> res_out;	// 输出结果的实数数据。

	cl::Buffer res_idx_buf;	// 输出结果的索引数据的缓冲区。
	cl::Buffer res_buf;	// 输出结果的实数数据的缓冲区。

	// Each partition has a fixed number of rows and a number of nnz that depends on the rows.
	// We track the first and last rows associated with the partition to split the total COO nnz;
	int_type num_rows_partition;	// 子SpMV分区的行数。
	int_type num_nnz_partition;	// 子SpMV分区的非零元素个数。
	int_type first_row;	// 子SpMV分区的第一个行的编号。
	int_type last_row;	// 子SpMV分区的最后一行的编号。

	int_type num_blocks_rows;	// 子SpMV分区的行块数。
	int_type num_blocks_nnz;	// 传输子SpMV分区需要的数据包个数

	Q_SubSpMVPartition(int_type _num_rows_partition, int_type _partition_id): num_rows_partition(_num_rows_partition), partition_id(_partition_id) {
		num_blocks_rows = (num_rows_partition + ULTRACSR_PACKET_SIZE - 1) / ULTRACSR_PACKET_SIZE;
		for (int_type i = 0; i < K * TOPK_RES_COPIES; i++) {
			input_packet_int_ultracsr packet_res_idx;
			input_packet_qtype_inout_ultracsr packet_res;
			res_idx_out.push_back(packet_res_idx);
			res_out.push_back(packet_res);
		}
	}
};

// 用于表示一个超级稀疏矩阵向量乘法（SpMV）分区。
struct SuperSpMVPartition {
	int_type partition_id;	// 超级稀疏矩阵的分区号。
	cl::Event write_event;	
	cl::Event reset_event;
	cl::Event computation_event;
	cl::Event readback_event;
	cl::Kernel kernel;	// 内核对象。
	cl::Buffer vec_buf;	// 输入向量的缓冲区。
	std::vector<SubSpMVPartition> partitions;	// 子SpMV分区的集合。

	SuperSpMVPartition(int_type _partition_id, std::vector<int_type> &_num_rows_partition, cl::Kernel _kernel): partition_id(_partition_id), kernel(_kernel) {
		for (int i = 0; i < SUB_SPMV_PARTITIONS; i++) {
			partitions.push_back(SubSpMVPartition(_num_rows_partition[i], partition_id * SUB_SPMV_PARTITIONS + i));
		}
	}
};

// 2024-05-22
struct Q_SuperSpMVPartition {
	int_type partition_id;	// 超级稀疏矩阵的分区号。
	cl::Event write_event;	
	cl::Event reset_event;
	cl::Event computation_event;
	cl::Event readback_event;
	cl::Kernel kernel;	// 内核对象。
	cl::Buffer vec_buf;	// 输入向量的缓冲区。
	std::vector<Q_SubSpMVPartition> q_partitions;	// 子SpMV分区的集合。

	Q_SuperSpMVPartition(int_type _partition_id, std::vector<int_type> &_num_rows_partition, cl::Kernel _kernel): partition_id(_partition_id), kernel(_kernel) {
		for (int i = 0; i < SUB_SPMV_PARTITIONS; i++) {
			q_partitions.push_back(Q_SubSpMVPartition(_num_rows_partition[i], partition_id * SUB_SPMV_PARTITIONS + i));
		}
	}
};

// 该结构体用于执行稀疏矩阵向量乘法（SpMV）的OpenCL加速计算。
// 结构体SpMV的主要功能是通过OpenCL加速执行稀疏矩阵向量乘法，
// 其中包括了数据的初始化、分块、内核设置、执行内核和结果读取等过程。这个结构体和相关函数组成了一个完整的稀疏矩阵向量乘法的加速计算流程，可以在FPGA上进行高效的计算。
struct SpMV {
	ConfigOpenCL config;

	int_type *x;
	int_type *y;
	real_type_inout* val;
	int_type *q_x;
	int_type *q_y;
	q_type* q_val;

	int_type num_rows;
	int_type num_cols;
	int_type num_nnz;
	int_type q_num_rows;
	int_type q_num_cols;
	int_type q_num_nnz;

	real_type_inout* vec;
	q_type* q_vec;
	vec_real_inout_bscsr* vec_in;
	vec_int_inout_ultracsr* q_vec_in;

	int_type num_blocks_cols;
	int_type q_num_blocks_cols;

	std::vector<SuperSpMVPartition> partitions;
	std::vector<Q_SuperSpMVPartition> q_partitions;

	// Keep a copy of events;
	std::vector<cl::Event> write_events;
	std::vector<cl::Event> reset_events;
	std::vector<cl::Event> computation_events;
	std::vector<cl::Event> readback_events;

	std::vector<cl::Event> q_write_events;
	std::vector<cl::Event> q_reset_events;
	std::vector<cl::Event> q_computation_events;
	std::vector<cl::Event> q_readback_events;

	// 初始化 全精度阶段 的环境
	SpMV(ConfigOpenCL& config_, int_type* x_, int_type* y_, real_type_inout* val_, int_type num_rows_, int_type num_cols_, int_type num_nnz_, real_type_inout* vec_, int debug = 0) :
			config(config_), x(x_), y(y_), val(val_), num_rows(num_rows_), num_cols(num_cols_), num_nnz(num_nnz_), vec(vec_) {

        // 计算用于存储输入向量分块的数组的大小，并分配内存空间；
        num_blocks_cols = (num_cols + BSCSR_PACKET_SIZE - 1) / BSCSR_PACKET_SIZE;
		posix_memalign((void**) &vec_in, 4096, num_blocks_cols * sizeof(vec_real_inout_bscsr));

		// Initialize partitions;   分成了8个SuperSpMVPartition，每个SuperSpMVPartition包含4个SubSpMVPartition，共计32个SpMV分区。
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			std::vector<int_type> num_rows_partitions;
			for (int_type j = 0; j < SUB_SPMV_PARTITIONS; j++) {
				int idx = i * SUB_SPMV_PARTITIONS + j;	// 计算当前SpMV分区的编号。（共32个分区）
				// Each partition has the same amount of rows, but the last partition might have fewer rows; 计算每个分区包含的SpM行数
				num_rows_partitions.push_back((idx == SPMV_PARTITIONS - 1) ? (num_rows / SPMV_PARTITIONS) : ((num_rows + SPMV_PARTITIONS - 1) / SPMV_PARTITIONS));
			}
			SuperSpMVPartition p(i, num_rows_partitions, config.kernel[i]);
			partitions.push_back(p);
		}
		// Create the COO data structure as a sequence of packets of (x, y, val);
		packet_coo(debug);
		setup(debug);
	}

	// 初始化 Quantize 近似阶段 的环境 2024-05-22
	SpMV(ConfigOpenCL &config_, int_type *q_x_, int_type *q_y_, q_type *q_val_, int_type q_num_rows_, int_type q_num_cols_, int_type q_num_nnz_, q_type *q_vec_, int debug=0) :
		config(config_), q_x(q_x_), q_y(q_y_), q_val(q_val_), q_num_rows(q_num_rows_), q_num_cols(q_num_cols_), q_num_nnz(q_num_nnz_), q_vec(q_vec_) {
		// 计算用于存储输入向量分块的数组的大小，并分配内存空间；
		q_num_blocks_cols = (q_num_cols + ULTRACSR_PACKET_SIZE - 1) / ULTRACSR_PACKET_SIZE;		// TODO：有bug
		posix_memalign((void**) &q_vec_in, 4096, q_num_blocks_cols * sizeof(vec_int_inout_ultracsr));

		// Initialize partitions;   分成了8个SuperSpMVPartition，每个SuperSpMVPartition包含4个SubSpMVPartition，共计32个SpMV分区。
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			std::vector<int_type> num_rows_partitions;
			for (int_type j = 0; j < SUB_SPMV_PARTITIONS; j++) {
				int idx = i * SUB_SPMV_PARTITIONS + j;	// 计算当前SpMV分区的编号。（共32个分区）
				// Each partition has the same amount of rows, but the last partition might have fewer rows; 计算每个分区包含的SpM行数
				num_rows_partitions.push_back((idx == SPMV_PARTITIONS - 1) ? (num_rows / SPMV_PARTITIONS) : ((num_rows + SPMV_PARTITIONS - 1) / SPMV_PARTITIONS));
			}
			Q_SuperSpMVPartition p(i, num_rows_partitions, config.kernel[i]);
			q_partitions.push_back(p);
		}
		// Create the q_COO data structure as a sequence of packets of (x, y, val);
		Q_packet_coo(debug);
		Q_setup(debug);
	}

	SubSpMVPartition& get_partition(int_type i) {
		int_type super = i / SUB_SPMV_PARTITIONS;
		int_type sub = i % SUB_SPMV_PARTITIONS;
		return partitions[super].partitions[sub];
	}

	// 2024-05-22
	Q_SubSpMVPartition& Q_get_partition(int_type i) {
		int_type super = i / SUB_SPMV_PARTITIONS;
		int_type sub = i % SUB_SPMV_PARTITIONS;
		return q_partitions[super].q_partitions[sub];
	}

	// 将COO格式的稀疏矩阵数据打包成适合于FPGA处理的格式
	void packet_coo(int debug = 0) {
		// First, split the COO; 通过计算该矩阵分成多少个分区，然后根据元素的行索引将每个元素分配到相应的分区中。
		int_type num_rows_per_partition = (num_rows + SPMV_PARTITIONS - 1) / SPMV_PARTITIONS;
		for (int_type i = 0; i < num_nnz; i++) {
			// Find the partition of this entry;
            int_type curr_p = x[i] / num_rows_per_partition;
            if (curr_p < SPMV_PARTITIONS)
                get_partition(curr_p).coo_partition.push_back(std::tuple<int_type, int_type, real_type_inout>(x[i], y[i], val[i]));
		}

		// Book-keeping; ，对每个分区进行一些记录，包括每个分区的首行、末行和非零元素数量，并计算每个分区需要多少个数据包来传输非零元素数据。
		for (int_type i = 0; i < SPMV_PARTITIONS; i++) {
			get_partition(i).first_row = std::get<0>(get_partition(i).coo_partition[0]);
			get_partition(i).last_row = std::get<0>(get_partition(i).coo_partition[get_partition(i).coo_partition.size() - 1]);
			get_partition(i).num_nnz_partition = get_partition(i).coo_partition.size();
			get_partition(i).num_blocks_nnz = (get_partition(i).num_nnz_partition + BSCSR_PACKET_SIZE - 1) / BSCSR_PACKET_SIZE;
			if (debug) std::cout << "partition " << i << ") nnz=" << get_partition(i).num_nnz_partition << "; blocks=" << get_partition(i).num_blocks_nnz << "; first_row=" << get_partition(i).first_row << "; last_row=" << get_partition(i).last_row << std::endl;
		}

		// Packet each COO partition; 对每个COO分区进行打包处理，打包成BS-CSR格式的数据数据包。
		int_type r_last = 0;
		for (int_type i = 0; i < SPMV_PARTITIONS; i++) {
			packet_coo_partition(get_partition(i), r_last);
			r_last = get_partition(i).last_row;
		}

		// 将输入向量也进行了分块处理，并存储到一个名为 vec_in 的数组中。
		real_type_inout vec_buffer[BSCSR_PACKET_SIZE];
		for (int_type i = 0; i < num_blocks_cols; i++) {
			vec_real_inout_bscsr new_block_vec;
			for (int_type j = 0; j < BSCSR_PACKET_SIZE; j++) {
				int_type index = j + BSCSR_PACKET_SIZE * i;
				if (index < num_cols) {
					vec_buffer[j] = vec[index];
				} else {
					vec_buffer[j] = 0;
				}
			}
			write_block_vec(&new_block_vec, vec_buffer);
			vec_in[i] = new_block_vec;
		}
	}

	// 将COO格式的稀疏矩阵数据打包成适合于FPGA处理的格式，2024-05-24
	void Q_packet_coo(int debug = 0) {
		// First, split the COO; 通过计算该矩阵分成多少个分区，然后根据元素的行索引将每个元素分配到相应的分区中。
		int_type num_rows_per_partition = (q_num_rows + SPMV_PARTITIONS - 1) / SPMV_PARTITIONS;
		for (int_type i = 0; i < q_num_nnz; i++) {
			// Find the partition of this entry;
			int_type curr_p = q_x[i] / num_rows_per_partition;
			if (curr_p < SPMV_PARTITIONS)
                Q_get_partition(curr_p).coo_partition.push_back(std::tuple<int_type, int_type, q_type>(q_x[i], q_y[i], q_val[i]));
		}
		std::cout<<"book-keeping q_partitions"<<std::endl;
		// Book-keeping; ，对每个分区进行一些记录，包括每个分区的首行、末行和非零元素数量，并计算每个分区需要多少个数据包来传输非零元素数据。
		for (int_type i = 0; i < SPMV_PARTITIONS; i++) {
			Q_get_partition(i).first_row = std::get<0>(Q_get_partition(i).coo_partition[0]);
			Q_get_partition(i).last_row = std::get<0>(Q_get_partition(i).coo_partition[Q_get_partition(i).coo_partition.size() - 1]);
			Q_get_partition(i).num_nnz_partition = Q_get_partition(i).coo_partition.size();
			Q_get_partition(i).num_blocks_nnz = (Q_get_partition(i).num_nnz_partition + ULTRACSR_PACKET_SIZE - 1) / ULTRACSR_PACKET_SIZE;
			if (debug) std::cout << "partition " << i << ") nnz=" << Q_get_partition(i).num_nnz_partition << "; blocks=" << Q_get_partition(i).num_blocks_nnz << "; first_row=" << Q_get_partition(i).first_row << "; last_row=" << Q_get_partition(i).last_row << std::endl;
		}

		// Packet each COO partition; 对每个COO分区进行打包处理，打包成BS-CSR格式的数据数据包。
		int_type r_last = 0;
		for (int_type i = 0; i < SPMV_PARTITIONS; i++) {
			Q_packet_coo_partition(Q_get_partition(i), r_last);
			r_last = Q_get_partition(i).last_row;
		}
		// 将输入向量也进行了分块处理，并存储到一个名为 q_vec_in 的数组中。
		q_type q_vec_buffer[ULTRACSR_PACKET_SIZE];
		for (int_type i = 0; i < q_num_blocks_cols; i++) {
			vec_int_inout_ultracsr new_block_vec;
			for (int_type j = 0; j < ULTRACSR_PACKET_SIZE; j++) {
				int_type index = j + ULTRACSR_PACKET_SIZE * i;
				if (index < q_num_cols) {
					q_vec_buffer[j] = q_vec[index];
				}
				else {
					q_vec_buffer[j] = 0;
				}
			}
			Q_write_block_vec(&new_block_vec, q_vec_buffer);
			q_vec_in[i] = new_block_vec;
		}
	}

	// 将COO格式转换为BS-CSR格式的数据包
	void packet_coo_partition(SubSpMVPartition& p, int_type last_r) {
		int_type curr_row = 0;
		int_type x_local[BSCSR_PACKET_SIZE];
		int_type y_local[BSCSR_PACKET_SIZE];
		real_type_inout val_local[BSCSR_PACKET_SIZE];
		bool_type xf_local[1];
		curr_row = last_r;
		// 对每个数据包进行打包处理
		for (int_type i = 0; i < p.num_blocks_nnz; i++) {
			input_block tmp_block_512;
			if (std::get<0>(p.coo_partition[BSCSR_PACKET_SIZE * i]) != curr_row){	// 数据包首个元素和上一数据包最后一个元素不是同行，则标记为新行
				xf_local[0] = (bool_type) true;
				write_block_xf(&tmp_block_512, xf_local);
				curr_row = std::get<0>(p.coo_partition[BSCSR_PACKET_SIZE * i]);
			} else {
				xf_local[0] = (bool_type) false;
				write_block_xf(&tmp_block_512, xf_local);
			}
			// 一个数据包打包 BSCSR_PACKET_SIZE 个nnz，打包y和val
			for (int_type j = 0; j < BSCSR_PACKET_SIZE; j++) {
				int_type index = j + BSCSR_PACKET_SIZE * i;
				auto curr_tuple = p.coo_partition[index];	// 读取第i个数据包的第j个元素（COO格式）
				curr_row = std::get<0>(curr_tuple);	// 当前行号
				if (index < p.num_nnz_partition) {
					x_local[j] = 0;
					y_local[j]= std::get<1>(curr_tuple);
					val_local[j] = std::get<2>(curr_tuple);
				} else {	// 没有元素了，往后补0️
					x_local[j] = 0;
					y_local[j] = 0;
					val_local[j] = 0;
				}
			}
			// 打包 x_ptr，将x坐标译码为x_ptr(x_local)
			int_type pos = 0;
			int_type same_row_values = 1;
			for (int_type j = 1; j < BSCSR_PACKET_SIZE; j++) {
				if (j - 1 + (BSCSR_PACKET_SIZE * i) < p.num_nnz_partition){
					if (std::get<0>(p.coo_partition[j + (BSCSR_PACKET_SIZE * i)]) == std::get<0>(p.coo_partition[j - 1 + (BSCSR_PACKET_SIZE * i)])) {
						same_row_values++;
					} else {
						x_local[pos] = same_row_values;
						same_row_values = 1;
						pos++;
					}
				} else {
					x_local[pos] = 0;
					pos++;
				}
			}
			if (BSCSR_PACKET_SIZE - 1 + (BSCSR_PACKET_SIZE * i) < p.num_nnz_partition){
				x_local[pos] = same_row_values;
			}
			// Accumulate values in x;
			for (int_type j = 1; j < BSCSR_PACKET_SIZE; j++) {
				x_local[j] += x_local[j - 1];
			}
			// 封装进一个数据包
			write_block_x(&tmp_block_512, x_local);
			write_block_y(&tmp_block_512, y_local);
			write_block_val(&tmp_block_512, val_local);
			p.bscsr_in.push_back(tmp_block_512);
		}
	}

	// 将COO格式转换为Ultra-CSR格式的数据包
	void Q_packet_coo_partition(Q_SubSpMVPartition& p, int_type last_partition_end_row) {
		int_type last_nnz_row = 0;
		int_type x_ptr_local[ULTRACSR_PACKET_SIZE];
		int_type y_local[ULTRACSR_PACKET_SIZE];
		q_type val_local[ULTRACSR_PACKET_SIZE];
		last_nnz_row = last_partition_end_row;
		// 对每个数据包进行打包处理
		for (int_type i = 0; i < p.num_blocks_nnz; i++) {
			q_input_block tmp_block_512;
			// 一个数据包打包 ULTRACSR_PACKET_SIZE 个nnz，打包y和val
			for (int_type j = 0; j < ULTRACSR_PACKET_SIZE; j++) {
				int_type index = j + ULTRACSR_PACKET_SIZE * i;
				auto curr_tuple = p.coo_partition[index];	// 读取第i个数据包的第j个元素（COO格式）
				int_type curr_row = std::get<0>(curr_tuple);	// 当前行号
				if (index < p.num_nnz_partition) {
					x_ptr_local[j] = curr_row==last_nnz_row ? 0 : 1;
					last_nnz_row = curr_row;
					y_local[j] = std::get<1>(curr_tuple);
					val_local[j] = std::get<2>(curr_tuple);
					// std::cout << "val_local: " << val_local[j] << std::endl; 
				}
				else {	// 没有元素了，往后补0️，可能存在bug
					x_ptr_local[j] = 0;
					y_local[j] = 0;
					val_local[j] = 0;
				}
			}
			// std::cout << i << "th block:" << std::endl;
			// for (int j = 0;j < ULTRACSR_PACKET_SIZE;j++) {
			// 	std::cout << "x_ptr_local[" << j << "]=" << x_ptr_local[j] << "\ty_local[" << j << "]=" << y_local[j] << "\tval_local[" << j << "]=" << val_local[j] << std::endl;
			// }
			// 封装到一个数据包中
			Q_write_block_x(&tmp_block_512, x_ptr_local);
			Q_write_block_y(&tmp_block_512, y_local);
			Q_write_block_val(&tmp_block_512, val_local);
			p.ultracsr_in.push_back(tmp_block_512);
		}
	}

	void setup(int debug) {

		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			if (debug) std::cout << "Create Kernel Arguments (partition " << i << ")" << std::endl;

			int a = i * SUB_SPMV_PARTITIONS + 0;
			int b = i * SUB_SPMV_PARTITIONS + 1;
			int c = i * SUB_SPMV_PARTITIONS + 2;
			int d = i * SUB_SPMV_PARTITIONS + 3;

			// Create the input and output arrays in device memory
			cl::Buffer coo_buf0 = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * get_partition(a).num_blocks_nnz, get_partition(a).bscsr_in.data());
			cl::Buffer coo_buf1 = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * get_partition(b).num_blocks_nnz, get_partition(b).bscsr_in.data());
			cl::Buffer coo_buf2 = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * get_partition(c).num_blocks_nnz, get_partition(c).bscsr_in.data());
			cl::Buffer coo_buf3 = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_block) * get_partition(d).num_blocks_nnz, get_partition(d).bscsr_in.data());

			partitions[i].vec_buf = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(vec_real_inout_bscsr) * num_blocks_cols, vec_in);
			get_partition(a).res_idx_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_int_bscsr) * K * TOPK_RES_COPIES, get_partition(a).res_idx_out.data());
			get_partition(b).res_idx_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_int_bscsr) * K * TOPK_RES_COPIES, get_partition(b).res_idx_out.data());
			get_partition(c).res_idx_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_int_bscsr) * K * TOPK_RES_COPIES, get_partition(c).res_idx_out.data());
			get_partition(d).res_idx_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_int_bscsr) * K * TOPK_RES_COPIES, get_partition(d).res_idx_out.data());
			get_partition(a).res_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_real_inout_bscsr) * K * TOPK_RES_COPIES, get_partition(a).res_out.data());
			get_partition(b).res_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_real_inout_bscsr) * K * TOPK_RES_COPIES, get_partition(b).res_out.data());
			get_partition(c).res_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_real_inout_bscsr) * K * TOPK_RES_COPIES, get_partition(c).res_out.data());
			get_partition(d).res_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_real_inout_bscsr) * K * TOPK_RES_COPIES, get_partition(d).res_out.data());

			// Set kernel arguments
			int narg = 0;
			partitions[i].kernel.setArg(narg++, coo_buf0);
			partitions[i].kernel.setArg(narg++, coo_buf1);
			partitions[i].kernel.setArg(narg++, coo_buf2);
			partitions[i].kernel.setArg(narg++, coo_buf3);

			partitions[i].kernel.setArg(narg++, get_partition(a).num_rows_partition);
			partitions[i].kernel.setArg(narg++, get_partition(b).num_rows_partition);
			partitions[i].kernel.setArg(narg++, get_partition(c).num_rows_partition);
			partitions[i].kernel.setArg(narg++, get_partition(d).num_rows_partition);

			partitions[i].kernel.setArg(narg++, num_cols);
			partitions[i].kernel.setArg(narg++, num_cols);
			partitions[i].kernel.setArg(narg++, num_cols);
			partitions[i].kernel.setArg(narg++, num_cols);

			partitions[i].kernel.setArg(narg++, get_partition(a).num_nnz_partition);
			partitions[i].kernel.setArg(narg++, get_partition(b).num_nnz_partition);
			partitions[i].kernel.setArg(narg++, get_partition(c).num_nnz_partition);
			partitions[i].kernel.setArg(narg++, get_partition(d).num_nnz_partition);

			partitions[i].kernel.setArg(narg++, partitions[i].vec_buf);

			partitions[i].kernel.setArg(narg++, get_partition(a).res_idx_buf);
			partitions[i].kernel.setArg(narg++, get_partition(b).res_idx_buf);
			partitions[i].kernel.setArg(narg++, get_partition(c).res_idx_buf);
			partitions[i].kernel.setArg(narg++, get_partition(d).res_idx_buf);

			partitions[i].kernel.setArg(narg++, get_partition(a).res_buf);
			partitions[i].kernel.setArg(narg++, get_partition(b).res_buf);
			partitions[i].kernel.setArg(narg++, get_partition(c).res_buf);
			partitions[i].kernel.setArg(narg++, get_partition(d).res_buf);

			// Transfer data from host to device (0 means host-to-device transfer);
			if (debug) std::cout << "Write inputs into device memory (partition " << i << ")" << std::endl;

			// Transfer data from host to device (0 means host-to-device transfer);
			config.queue[i % OPENCL_QUEUES].enqueueMigrateMemObjects( { coo_buf0, coo_buf1, coo_buf2, coo_buf3, partitions[i].vec_buf }, 0, NULL, &partitions[i].write_event);
			write_events.push_back(partitions[i].write_event);

		}
		// Wait for completion of transfer;
		cl::Event::waitForEvents(write_events);
		write_events.clear();
	}

	// TODO：
	void Q_setup(int debug) {
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			if (debug) std::cout << "Create Kernel Arguments (partition " << i << ")" << std::endl;
			int a = i * SUB_SPMV_PARTITIONS + 0;

			// Create the input and output arrays in device memory
			cl::Buffer coo_buf0 = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(q_input_block) * Q_get_partition(a).num_blocks_nnz, Q_get_partition(a).ultracsr_in.data());

			q_partitions[i].vec_buf = cl::Buffer(config.context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(vec_int_inout_ultracsr) * q_num_blocks_cols, q_vec_in);
			Q_get_partition(a).res_idx_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_int_ultracsr) * K * TOPK_RES_COPIES, Q_get_partition(a).res_idx_out.data());
			Q_get_partition(a).res_buf = cl::Buffer(config.context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(input_packet_qtype_inout_ultracsr) * K * TOPK_RES_COPIES, Q_get_partition(a).res_out.data());

			// Set kernel arguments
			int narg = 0;
			q_partitions[i].kernel.setArg(narg++, coo_buf0);

			q_partitions[i].kernel.setArg(narg++, Q_get_partition(a).num_rows_partition);

			q_partitions[i].kernel.setArg(narg++, q_num_cols);

			q_partitions[i].kernel.setArg(narg++, Q_get_partition(a).num_nnz_partition);

			q_partitions[i].kernel.setArg(narg++, q_partitions[i].vec_buf);

			q_partitions[i].kernel.setArg(narg++, Q_get_partition(a).res_idx_buf);

			q_partitions[i].kernel.setArg(narg++, Q_get_partition(a).res_buf);

			// Transfer data from host to device (0 means host-to-device transfer);
			if (debug) std::cout << "Write inputs into device memory (partition " << i << ")" << std::endl;

			// Transfer data from host to device (0 means host-to-device transfer);
			config.queue[i % OPENCL_QUEUES].enqueueMigrateMemObjects( { coo_buf0, q_partitions[i].vec_buf }, 0, NULL, &q_partitions[i].write_event);
			q_write_events.push_back(q_partitions[i].write_event);
		}
		// Wait for completion of transfer;
		cl::Event::waitForEvents(q_write_events);
		q_write_events.clear();
	}

	// 一个重载的函数调用操作符，它用于执行FPGA上的计算。
	// 它通过执行FPGA上的计算内核来执行超级稀疏矩阵向量乘法（SpMV）计算，返回计算所用的时间
	// 该函数遍历超级SpMV分区的集合并执行每个分区对应的计算内核。它通过执行FPGA上的计算内核来执行超级稀疏矩阵向量乘法（SpMV）计算。该函数遍历超级SpMV分区的集合并执行每个分区对应的计算内核。
	long operator()(int debug) {
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			if (debug)  std::cout << "Execute the kernel (partition " << i << ")" << std::endl;
//			cl::Event e; 它使用OpenCL队列的enqueueTask方法来执行每个超级SpMV分区对应的计算内核，并将计算事件添加到computation_events集合中。
			config.queue[i % OPENCL_QUEUES].enqueueTask(partitions[i].kernel, NULL, &partitions[i].computation_event);
			computation_events.push_back(partitions[i].computation_event);
		}
		// Wait for computation to end;
		return wait(debug);
	}

	// 一个重载的函数调用操作符，它用于执行FPGA上的计算。执行 Quantized SpMV 计算，返回计算所用的时间。
	long operator()(bool flag, int debug) {
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			if (debug)  std::cout << "Execute the kernel (partition " << i << ")" << std::endl;
//			cl::Event e; 它使用OpenCL队列的enqueueTask方法来执行每个超级SpMV分区对应的计算内核，并将计算事件添加到computation_events集合中。
			config.queue[i % OPENCL_QUEUES].enqueueTask(q_partitions[i].kernel, NULL, &q_partitions[i].computation_event);
			q_computation_events.push_back(q_partitions[i].computation_event);
		}
		// Wait for computation to end;
		return Q_wait(debug);
	}

	// 等待计算事件结束，读取计算结果，并返回计算时间，用于分析性能。
	long wait(int debug) {
		auto start = clock_type::now();
		// Wait for completion of computation and read-back;
		cl::Event::waitForEvents(computation_events);
		auto elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();
		if (debug) {
			std::cout << "Kernel terminated" << std::endl;
			std::cout << "Computation took " << elapsed / 1e6 << " ms" << std::endl;
		}

		// Read-back; 读取计算结果
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			if (debug)  std::cout << "Read result from kernel (partition " << i << ")" << std::endl;

			int a = i * SUB_SPMV_PARTITIONS + 0;

			// Read back the results from the device to verify the output
			config.queue[i % OPENCL_QUEUES].enqueueMigrateMemObjects({
				get_partition(a).res_idx_buf,
				get_partition(a).res_buf,
			}, CL_MIGRATE_MEM_OBJECT_HOST, 0, &partitions[i].readback_event);
			readback_events.push_back(partitions[i].readback_event);
		}
		start = clock_type::now();
		cl::Event::waitForEvents(readback_events);
		auto read_elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();

		if (debug) {
			std::cout << "Read-back took " << read_elapsed / 1e6 << " ms" << std::endl;
		}

		readback_events.clear();
		computation_events.clear();
		return elapsed;
	}

		// 等待计算事件结束，读取计算结果，并返回计算时间，用于分析性能。
	long Q_wait(int debug) {
		auto start = clock_type::now();
		// Wait for completion of computation and read-back;
		cl::Event::waitForEvents(q_computation_events);
		auto elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();
		if (debug) {
			std::cout << "Kernel terminated" << std::endl;
			std::cout << "Computation took " << elapsed / 1e6 << " ms" << std::endl;
		}
		// Read-back; 读取计算结果
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			if (debug)  std::cout << "Read result from kernel (partition " << i << ")" << std::endl;
			int a = i * SUB_SPMV_PARTITIONS + 0;
			// Read back the results from the device to verify the output
			config.queue[i % OPENCL_QUEUES].enqueueMigrateMemObjects({
				Q_get_partition(a).res_idx_buf,
				Q_get_partition(a).res_buf,
			}, CL_MIGRATE_MEM_OBJECT_HOST, 0, &q_partitions[i].readback_event);
			q_readback_events.push_back(q_partitions[i].readback_event);
		}
		start = clock_type::now();
		cl::Event::waitForEvents(q_readback_events);
		auto read_elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();
		if (debug) {
			std::cout << "Read-back took " << read_elapsed / 1e6 << " ms" << std::endl;
		}
		q_readback_events.clear();
		q_computation_events.clear();
		return elapsed;
	}

	void read_result(std::vector<real_type_inout> &res, std::vector<int_type> &res_idx, int debug=0) {

		// Store results in a map;
		std::unordered_map<int_type, real_type_inout> result_map;

		// Read output of each partition;
		for (int i = 0; i < SPMV_PARTITIONS; i++) {
			// Temporary vectors to store results for the current partition, used only for debug;
			std::vector<int_type> res_idx_tmp;
			std::vector<real_type_inout> res_tmp;

			for (int_type c = 0; c < TOPK_RES_COPIES; c++) {
				for (int_type j = 0; j < K; j++) {
					for (int_type q = 0; q < BSCSR_PACKET_SIZE; q++) {
						int_type index = q + BSCSR_PACKET_SIZE * (j + K * c);
						int_type index_2 = j + c * K;
						int_type tmp_idx = get_partition(i).res_idx_out[index_2][q] + get_partition(i).first_row; //We need to to add the starting row of the partition
						real_type_inout tmp_val = get_partition(i).res_out[index_2][q];
						if (tmp_val > 0) {  // Skip empty results;
							result_map.insert(std::pair<int_type, real_type_inout>(tmp_idx, tmp_val));
						}

						// Store results for each partition;
						if (debug) {
							res_idx_tmp.push_back(tmp_idx);
							res_tmp.push_back(tmp_val);
						}
					}
				}
			}

			// if (debug) {
			// 	sort_tuples(res_idx_tmp.size(), res_idx_tmp.data(), res_tmp.data());
			// 	std::cout << "\nPartition " << i << " results:" << std::endl;
			// 	for (int_type j = 0; j < res_idx_tmp.size(); j++) {
			// 		if (res_tmp[j] == 0) {
			// 			break;
			// 		}
			// 		std::cout << j << ") " << res_idx_tmp[j] << "=" << res_tmp[j] << std::endl;
			// 	}
			// }
		}

		int_type i = 0;
		for(auto it = result_map.begin(); it != result_map.end(); it++, i++) {
			res_idx.push_back(it->first);
			res.push_back(it->second);
		}
		auto start = clock_type::now();
		sort_tuples(res.size(), res_idx.data(), res.data());	// 读取计算结果并排序，这个时延为0.1ms左右
		auto end = clock_type::now();
		float sort_time = (float)chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000;	// 计算真实SpMV的运行时间
		std::cout << "Sort time: " << sort_time << " ms" << std::endl;
	}

	void Q_read_result(std::vector<q_type_inout> &res, std::vector<int_type> &res_idx, int debug=0) {

		// Store results in a map;
		std::unordered_map<int_type, q_type_inout> result_map;

		// Read output of each partition;
		for (int i = 0; i < SPMV_PARTITIONS; i++) {
			// Temporary vectors to store results for the current partition, used only for debug;
			std::vector<int_type> res_idx_tmp;
			std::vector<q_type_inout> res_tmp;

			for (int_type c = 0; c < TOPK_RES_COPIES; c++) {
				for (int_type j = 0; j < K; j++) {
					for (int_type q = 0; q < Q_LIMITED_FINISHED_ROWS; q++) {
						int_type index = q + Q_LIMITED_FINISHED_ROWS * (j + K * c);
						int_type index_2 = j + c * K;
						int_type tmp_idx = Q_get_partition(i).res_idx_out[index_2][q] + Q_get_partition(i).first_row; //We need to to add the starting row of the partition
						q_type_inout tmp_val = Q_get_partition(i).res_out[index_2][q];
						if (tmp_val > 0) {  // Skip empty results;
							result_map.insert(std::pair<int_type, q_type_inout>(tmp_idx, tmp_val));
						}

						// Store results for each partition;
						if (debug) {
							res_idx_tmp.push_back(tmp_idx);
							res_tmp.push_back(tmp_val);
						}
					}
				}
			}
			// if (debug) {
			// 	sort_tuples(res_idx_tmp.size(), res_idx_tmp.data(), res_tmp.data());
			// 	std::cout << "\nPartition " << i << " results:" << std::endl;
			// 	for (int_type j = 0; j < res_idx_tmp.size(); j++) {
			// 		if (res_tmp[j] == 0) {
			// 			break;
			// 		}
			// 		std::cout << j << ") " << res_idx_tmp[j] << "=" << res_tmp[j] << std::endl;
			// 	}
			// }
		}

		int_type i = 0;
		for(auto it = result_map.begin(); it != result_map.end(); it++, i++) {
			res_idx.push_back(it->first);
			res.push_back(it->second);
		}
		auto start = clock_type::now();
		sort_tuples(res.size(), res_idx.data(), res.data());	// 读取计算结果并排序，这个时延为0.1ms左右
		// std::cout << "Sort Results:" << std::endl;
		// std::cout << "res.size()" << res.size() << std::endl;
		// for (int i = 0;i < res.size();i++) {
		// 	std::cout << res_idx[i] << " " << res[i] << std::endl;
		// }
		auto end = clock_type::now();
		float sort_time = (float)chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000;	// 计算真实SpMV的运行时间
		std::cout << "Sort time: " << sort_time << " ms" << std::endl;
	}

	// 用于重新设置输入向量并将其加载到FPGA上。
	// 参数:
	//   - vec: 新的输入向量
	// 返回值:
	//   - 运行时间
	long reset(real_type_inout* vec_, int debug) {
		auto start = clock_type::now();
		vec = vec_;
		// Packet the input vector;
		real_type_inout vec_buffer[BSCSR_PACKET_SIZE];
		for (int_type i = 0; i < num_blocks_cols; i++) {
			vec_real_inout_bscsr new_block_vec;
			for (int_type j = 0; j < BSCSR_PACKET_SIZE; j++) {
				int_type index = j + BSCSR_PACKET_SIZE * i;
				if (index < num_cols) {
					vec_buffer[j] = vec[index];
				} else {
					vec_buffer[j] = 0;
				}
			}
			write_block_vec(&new_block_vec, vec_buffer);
			vec_in[i] = new_block_vec;
		}

		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			// Transfer data from host to device (0 means host-to-device transfer);
			config.queue[i % OPENCL_QUEUES].enqueueMigrateMemObjects( { partitions[i].vec_buf }, 0, NULL, &partitions[i].reset_event);
			reset_events.push_back(partitions[i].reset_event);
		}
		// Wait for completion of transfer;
		cl::Event::waitForEvents(reset_events);
		reset_events.clear();
		auto elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();
		if (debug) {
			std::cout << "Reset took " << elapsed / 1e6 << " ms" << std::endl;
		}
		return elapsed;
	}

	// 用于重新设置 Quantized 输入向量并将其加载到FPGA上。
	long Q_reset(q_type* vec_, int debug) {
		auto start = clock_type::now();
		q_vec = vec_;
		// Packet the input vector;
		q_type vec_buffer[ULTRACSR_PACKET_SIZE];
		for (int_type i = 0; i < q_num_blocks_cols; i++) {	// num_cols=512时，num_blocks_cols=18
			vec_int_inout_ultracsr new_block_vec;
			for (int_type j = 0; j < ULTRACSR_PACKET_SIZE; j++) {	// 30
				int_type index = j + ULTRACSR_PACKET_SIZE * i;
				if (index < q_num_cols) {
					vec_buffer[j] = q_vec[index];
				} else {
					vec_buffer[j] = 0;
				}
			}
			// std::cout << "Input vector block " << i << ":" << std::endl;
			// for (int_type j = 0; j < ULTRACSR_PACKET_SIZE; j++) {
				// std::cout << vec_buffer[j] << " ";
			// }
			// std::cout << std::endl;
			Q_write_block_vec(&new_block_vec, vec_buffer);
			q_vec_in[i] = new_block_vec;
		}
		for (int_type i = 0; i < SUPER_SPMV_PARTITIONS; i++) {
			// Transfer data from host to device (0 means host-to-device transfer);
			config.queue[i % OPENCL_QUEUES].enqueueMigrateMemObjects( { q_partitions[i].vec_buf }, 0, NULL, &q_partitions[i].reset_event);
			q_reset_events.push_back(q_partitions[i].reset_event);
		}

		// Wait for completion of transfer;
		cl::Event::waitForEvents(q_reset_events);
		q_reset_events.clear();

		auto elapsed = chrono::duration_cast<chrono::nanoseconds>(clock_type::now() - start).count();
		if (debug) {
			std::cout << "Reset took " << elapsed / 1e6 << " ms" << std::endl;
		}
		return elapsed;
	}
};

// 计算SW测试的函数
// 参数:
//   - coo: coo格式稀疏矩阵SpM
//   - sw_res: SpMV的真实结果值
//   - res_idx_sw: 软仿的SpMV的结果行索引
//   - res_sim_sw: 软仿的SpMV的结果值
//   - vec: 稠密向量V
//   - top_k_value: k值
// 返回值:
//   - 成功则返回测试时间
template<typename I, typename V>
std::tuple<float, float> sw_test(coo_t<I, V> &coo, std::vector<V> &sw_res, std::vector<int_type> &sw_res_idx, std::vector<V> &sim_res, std::vector<int_type> &sim_res_idx, V *vec, int_type top_k_value) {
	auto start_2 = clock_type::now();
	spmv_coo_gold4(coo, sw_res.data(), vec);	// 计算真实SpMV的结果
	std::vector<int_type> sw_res_idx_all = sort_pr(sw_res.size(), sw_res.data());	// 对真实SpMV的结果进行排序
	sw_res_idx.assign(sw_res_idx_all.begin(), sw_res_idx_all.begin() + top_k_value);	// 取前k个结果的索引
	auto end_2 = clock_type::now();
	float sw_time_1 = (float) chrono::duration_cast<chrono::microseconds>(end_2 - start_2).count() / 1000;	// 计算真实SpMV的运行时间

	auto start_3 = clock_type::now();
	spmv_coo_gold_top_k(coo, vec, top_k_value, sim_res_idx.data(), sim_res.data());	// 计算软仿SpMV的结果
	sort_tuples(top_k_value, sim_res_idx.data(), sim_res.data());	// 对软仿SpMV的结果进行排序
	auto end_3 = clock_type::now();
	float sw_time_2 = (float) chrono::duration_cast<chrono::microseconds>(end_3 - start_3).count() / 1000;

	return std::make_tuple(sw_time_1, sw_time_2);
}

// 计算量化后的SW测试的函数
// 参数:
//   - coo: coo格式稀疏矩阵SpM
// 	 - pcoo: 量化后的稀疏矩阵SpM
//   - q_vec: 量化后的向量qv
//   - sw_res: SpMV的真实结果值
//   - res_idx_sw: 软仿的SpMV的结果行索引
//   - res_sim_sw: 软仿的SpMV的结果值
//   - vec: 稠密向量V
//   - top_k_value: k值
// 返回值:
//   - 成功则返回测试时间
template<typename I, typename V, typename W, typename L>
std::tuple<float, float> q_sw_test(coo_t<I, V>& coo, coo_t<I, W>& q_coo, std::vector<V>& sw_res, std::vector<int_type>& sw_res_idx,
	std::vector<L>& sim_res, std::vector<int_type>& sim_res_idx, V* vec, W* q_vec, int_type top_k_value) {
	
	auto start_2 = clock_type::now();
	spmv_coo_gold4(coo, sw_res.data(), vec);	// 计算真实SpMV的结果，计算Top-100
	std::vector<int_type> sw_res_idx_all = sort_pr(sw_res.size(), sw_res.data());	// 对真实SpMV的结果进行排序
	sw_res_idx.assign(sw_res_idx_all.begin(), sw_res_idx_all.begin() + top_k_value);	// 取前k个结果
	auto end_2 = clock_type::now();
	float sw_time_1 = (float) chrono::duration_cast<chrono::microseconds>(end_2 - start_2).count() / 1000;	// 计算真实SpMV的运行时间
	auto start_3 = clock_type::now();
	spmv_coo_gold_top_k_2(q_coo, q_vec, APPROXIMATE_K, sim_res_idx.data(), sim_res.data());	// 计算软仿SpMV的结果，计算Top-512
	sort_tuples(top_k_value, sim_res_idx.data(), sim_res.data());	// 对软仿SpMV的结果进行排序
	auto end_3 = clock_type::now();
	float sw_time_2 = (float) chrono::duration_cast<chrono::microseconds>(end_3 - start_3).count() / 1000;
	return std::make_tuple(sw_time_1, sw_time_2);
}


/////////////////////////////
//     主函数				//
/////////////////////////////

int main(int argc, char *argv[]) {
    // 解析命令行参数
	Options options = Options(argc, argv);
	int debug = (int) options.debug;
	bool reset = options.reset;
	int seed = 0;  // If 0, don't use seed;
	cl::Context context;    // OpenCL上下文

	/////////////////////////////////
	//       Setup OpenCL        ////
	/////////////////////////////////
    // 指定目标设备、编译选项、xclbin路径、kernel名称
    std::string xclbin_path = options.xclbin_path;
	std::vector<std::string> target_devices = { "xilinx_u280_gen3x16_xdma_1_202211_1" };
	std::vector<std::string> kernels = { xclbin_path };
	std::string kernel_name = "spmv_ucsr_top_k_main";
	// setup kernel（配置OpenCL内核）
	// ConfigOpenCL config(kernel_name, SUPER_SPMV_PARTITIONS, OPENCL_QUEUES);	// 配置OpenCL内核的名称、分区数、并行队列数
	// setup_opencl(config, target_devices, kernels, debug);	// 根据options.xclbin_path和target_devices 创建（启动） OpenCL Kernels
	int top_k_value = options.top_k_value;

	std::cout << "before set up" <<  std::endl;
	/////////////////////////////////
	//    Setup Quantize OpenCL  ////
	/////////////////////////////////
	// std::vector<std::string> kernels = { xclbin_path };	// 编辑完量化的kernel后可能要改
	// std::string kernel_name = "spmv_bscsr_top_k_main";
	// setup kernel（配置OpenCL内核）
	ConfigOpenCL q_config(kernel_name, SUPER_SPMV_PARTITIONS, OPENCL_QUEUES);	// 配置OpenCL内核的名称、分区数、并行队列数
	setup_opencl(q_config, target_devices, kernels, debug);	// 根据options.xclbin_path和target_devices 创建（启动） OpenCL Kernels
	std::cout << "set up" <<  std::endl;

	/////////////////////////////////
	//     Load SpM & Vector       //
	/////////////////////////////////
	int_type nnz;
	int_type rows;
	int_type cols; // Size of the dense vector multiplied by the matrix;
	std::vector<int_type> x;    // x 和 y 是非负整数型
	std::vector<int_type> y;
	std::vector<real_type_inout> val;   // val 是无符号浮点型，整数位只有一位（只能表示0-1之间的浮点数）
	int read_values = !options.ignore_matrix_values; // If false, all values in the matrix are = 1; Set it true only for non-graphs;
	auto start_1 = clock_type::now();
	// 这里读出来的还是COO格式
	readMtx(options.use_sample_matrix ? DEFAULT_MTX_FILE : options.matrix_path.c_str(), &x, &y, &val, &rows, &cols, &nnz, 0, read_values, debug, true, false);	// 读出来的有正有负
	// Wrap the COO matrix;
    // 将读取的COO格式稀疏矩阵转换用COO结构体包裹，依旧是COO格式
	coo_t<int_type, real_type_inout> coo = coo_t<int_type, real_type_inout>(x, y, val);
	// coo.print_coo(true);
	// Vector multiplied by the sparse matrix; 生成稠密向量vec
	real_type_inout *vec;
	posix_memalign((void**)&vec, 4096, cols * sizeof(real_type_inout));    // 返回size字节的动态内存，内存地址是alignment（4096）的倍数，内存对齐是为了提高效率
	create_sample_vector(vec, cols, true, false, false, seed);	// 生成用于软件测试的随机向量，随机向量值随机（-1,1）
	std::cout << "Random vector:" << std::endl;
	print_array_indexed(vec, cols);
	auto end_1 = clock_type::now();
	auto loading_time = chrono::duration_cast<chrono::milliseconds>(end_1 - start_1).count();
	if (debug) {
		std::cout << "loaded matrix with " << rows << " rows, " << cols << " columns and " << nnz << " non-zero elements" << std::endl;
		std::cout << "setup time=" << loading_time << " ms" << std::endl;
	}


	/////////////////////////////////
	// Load Quantified SpM and QV  //
	/////////////////////////////////

	int_type qnnz;
	int_type qrows;
	int_type qcols; // Size of the dense vector multiplied by the matrix;
	std::vector<int_type> qx;    // x 和 y 是非负整数型
	std::vector<int_type> qy;
	std::vector<q_type> qval;   // val 是无符号浮点型，整数位只有一位（只能表示0-1之间的浮点数）
	auto start_1_q = clock_type::now();
	// 这里读出来的还是COO格式
	readMtx(options.q_matrix_path.c_str(), &qx, &qy, &qval, &qrows, &qcols, &qnnz, 0, read_values, debug, true, false);	// 读出来的有正有负
	// Wrap the COO matrix;
	coo_t<int_type, q_type> q_coo = coo_t<int_type, q_type>(qx, qy, qval);     // 将读取的COO格式稀疏矩阵转换用COO结构体包裹，依旧是COO格式
	// q_coo.print_coo(true);
	// 计算量化后的向量qv
	q_type* q_vec;
	posix_memalign((void**)&q_vec, 4096, qrows * sizeof(q_type));    // 返回size字节的动态内存，内存地址是alignment（4096）的倍数，内存对齐是为了提高效率
	auto start_2_q = clock_type::now();
	q_vec = quantize_vec(vec, qcols, QUANTIZE_B);
	print_array_indexed(q_vec, qcols);
	auto end_1_q = clock_type::now();
	auto loading_time_q = chrono::duration_cast<chrono::milliseconds>(end_1_q - start_1_q).count();
	auto quantization_time = chrono::duration_cast<chrono::milliseconds>(end_1_q - start_2_q).count();
	if (debug) {
		std::cout << "loaded Q_matrix with " << qrows << " rows, " << qcols << " columns and " << qnnz << " non-zero elements" << std::endl;
		std::cout << "setup time=" << loading_time_q << " ms" << std::endl;
		std::cout << "quantize vector time=" << quantization_time << " ms" << std::endl;
	}
	
	//////////////////////////////
	// Generate software result //
	//////////////////////////////

	// Output of software SpMV, it contains all the similarities for all documents;
	std::vector<real_type_inout> sw_res(coo.num_rows, 0);
	std::vector<int_type> sw_res_idx(coo.num_rows, 0);
	std::vector<real_type_inout> sim_res(top_k_value, 0);
	std::vector<int_type> sim_res_idx(top_k_value, 0);
	std::tuple<float, float> sw_time = sw_test(coo, sw_res, sw_res_idx, sim_res, sim_res_idx, vec, top_k_value);	// 计算软件SpMV的运行时间和结果
	float sw_time_1 = std::get<0>(sw_time);
	float sw_time_2 = std::get<1>(sw_time);
	if (debug) {
		std::cout << "\nsw top-100 results =" << std::endl;
		for (int i = 0; i < top_k_value; i++) {
			std::cout << i << ") document " << sim_res_idx[i] << " = " << sim_res[i] << std::endl;
		}
		std::cout << "sw errors = " << check_array_equality(sw_res.data(), sim_res.data(), 10e-6, top_k_value, true) << std::endl;
		std::cout << "sw time, full matrix=" << sw_time_1 << " ms; sw time, top-k=" << sw_time_2 << " ms" << std::endl;

		std::unordered_set<int> s(sw_res_idx.begin(), sw_res_idx.end());
		int_type size_intersection = count_if(sim_res_idx.begin(), sim_res_idx.end(), [&](int k) {return s.find(k) != s.end();});	// 返回了hw_res_idx和res_idx_sw中相同元素的数量。
		float precision = (((float)size_intersection) / ((float)top_k_value));
		std::cout << "Software Simulation Precision=" << precision * 100 << "%\n" << std::endl;
	}


	///////////////////////////////////
	// Generate Quantified sw result //
	///////////////////////////////////

	std::vector<q_type_inout> q_sim_res(APPROXIMATE_K, 0);
	std::vector<int_type> q_sim_res_idx(APPROXIMATE_K, 0);
	std::tuple<float, float> q_sw_time = q_sw_test(coo, q_coo, sw_res, sw_res_idx, q_sim_res, q_sim_res_idx, vec, q_vec, top_k_value);
	float q_sw_time_1 = std::get<0>(q_sw_time);
	float q_sw_time_2 = std::get<1>(q_sw_time);
	float precision = 0;

	if (debug) {
		std::cout << "\nquantized top-512 results =" << std::endl;
		for (int i = 0; i < APPROXIMATE_K; i++) {
			std::cout << i << ") document " << q_sim_res_idx[i] << " = " << q_sim_res[i] << std::endl;
		}
		std::cout << "sw time, full matrix=" << q_sw_time_1 << " ms; sw time, top-k=" << q_sw_time_2 << " ms" << std::endl;
		std::unordered_set<int> s(sw_res_idx.begin(), sw_res_idx.end());
		// 用Top-512量化近似真实Top-100
		int_type size_intersection = count_if(q_sim_res_idx.begin(), q_sim_res_idx.end(), [&](int k) {return s.find(k) != s.end();});
		precision = (((float)size_intersection) / ((float)top_k_value));
		// std::cout << "sw_res_idx.size() = " << sw_res_idx.size() << std::endl;
		// for (int i = 0; i < sw_res_idx.size(); i++) {
		// 	std::cout << sw_res_idx[i] << ",";
		// }
		// std::cout << "\nq_sim_res_idx.size() = " << q_sim_res_idx.size() << std::endl;
		// for (int i = 0; i < q_sim_res_idx.size(); i++) {
		// 	std::cout << q_sim_res_idx[i] << ",";
		// }
		// std::cout << std::endl;
		std::cout << "Quantized Software Simulation Precision=" << precision * 100 << "%\n" << std::endl;
	}
	
	/////////////////////////////
	// Setup hardware ///////////
	/////////////////////////////

	// auto start_4 = clock_type::now();
	// SpMV spmv(config, coo.start.data(), coo.end.data(), coo.val.data(), rows, cols, nnz, vec, debug);	// 实例化SpMV类，将数据分块加载到FPGA上
	// auto end_4 = clock_type::now();
	// float fpga_setup_time = (float) chrono::duration_cast<chrono::microseconds>(end_4 - start_4).count() / 1000;
	// if (debug) {
	// 	std::cout << "fpga setup time=" << fpga_setup_time << " ms" << std::endl;
	// }


	//////////////////////////////
	// Setup Quantized hardware //
	//////////////////////////////

	auto start_4_q = clock_type::now();
	SpMV q_spmv(q_config, q_coo.start.data(), q_coo.end.data(), q_coo.val.data(), qrows, qcols, qnnz, q_vec, debug);
	auto end_4_q = clock_type::now();
	float q_fpga_setup_time = (float) chrono::duration_cast<chrono::microseconds>(end_4_q - start_4_q).count() / 1000;
	if (debug) {
		std::cout << "quantized fpga setup time=" << q_fpga_setup_time << " ms" << std::endl;
	}

	
	/////////////////////////////
	// Execute the kernel ///////
	/////////////////////////////

	uint num_tests = options.num_tests;
	std::vector<float> exec_times_full;
	std::vector<float> exec_times;
	std::vector<float> readback_times;
	std::vector<float> error_count;
	std::vector<float> precision_vec;
	for (uint i = 0; i < num_tests; i++) {
		if (debug) {
			std::cout << "\nIteration " << i << ")" << std::endl;
		}
        // Create a new input vector;
        if (reset) {
			create_sample_vector(vec, cols, true, false, false, seed);
			// print_array_indexed(vec, cols);
			q_vec = quantize_vec(vec, qcols, QUANTIZE_B);
			print_array_indexed(q_vec, qcols);
			// 调用sw_test函数来进行软件仿真测试，该函数计算COO格式稀疏矩阵与输入向量的乘积，并返回两个时间值sw_time_1和sw_time_2
			std::tuple<float, float> q_sw_time = q_sw_test(coo, q_coo, sw_res, sw_res_idx, q_sim_res, q_sim_res_idx, vec, q_vec, top_k_value);
			q_sw_time_1 = std::get<0>(q_sw_time);
			q_sw_time_2 = std::get<1>(q_sw_time);
			// Load the new vec on FPGA;
			q_spmv.Q_reset(q_vec, debug); // 在FPGA上加载新的输入向量vec并进行初始化。
		}
		// Final output of hardware SpMV, it contains only the Top-K similarities and the Top-K indices;
		std::vector<q_type_inout> hw_res;
		std::vector<int_type> hw_res_idx;
		// Main FPGA computation;
		auto start_5 = clock_type::now();
		float fpga_exec_time = (float) q_spmv(true, debug) / 1e6;	// 调用SpMV类的operator()函数，执行FPGA上的计算，并返回运行时间
		auto end_5 = clock_type::now();
		float fpga_full_exec_time = (float) chrono::duration_cast<chrono::nanoseconds>(end_5 - start_5).count() / 1e6;
		exec_times.push_back(fpga_exec_time);
		exec_times_full.push_back(fpga_full_exec_time);
		// Retrieve results;
		auto start_6 = clock_type::now();
		q_spmv.Q_read_result(hw_res, hw_res_idx, debug);
		auto end_6 = clock_type::now();
		float readback_time = (float) chrono::duration_cast<chrono::microseconds>(end_6 - start_6).count() / 1000;
		readback_times.push_back(readback_time);

		//////////////////////////////
		// Check correctness /////////
		//////////////////////////////
		int res_size = (int)hw_res_idx.size();
		int error_idx = check_array_equality(hw_res_idx.data(), sw_res_idx.data(), std::min(top_k_value, res_size), 0, debug);
		error_count.push_back(error_idx);
		error_idx += std::max(0, top_k_value - res_size);
		std::unordered_set<int> s(sw_res_idx.begin(), sw_res_idx.end());
		// 计算真实top-100 idx 落入近似的 top-512 idx 中的比例（Precision）
		int_type size_intersection = count_if(q_sim_res_idx.begin(), q_sim_res_idx.end(), [&](int k) {return s.find(k) != s.end();});	// 返回了hw_res_idx和res_idx_sw中相同元素的数量。
		precision = (((float)size_intersection) / ((float)top_k_value));
		// std::cout << "precision=" << precision * 100 << "%" << std::endl;
		precision_vec.push_back(precision);

		if (debug) {
			// std::cout << "sw results =" << std::endl;
			// for (int j = 0; j < top_k_value; j++) {
			// 	std::cout << j << ") document " << res_idx_sw[j] << " = " << res_sim_sw[j] << std::endl;
			// }
			// std::cout << "hw results=" << std::endl;
			// for (int j = 0; j < std::min(top_k_value, res_size); j++) {
			// 	std::cout << j << ") document " << hw_res_idx[j] << " = " << hw_res[j] << std::endl;
			// }
			// std::cout << "num errors on indices=" << error_idx << std::endl;
			std::cout << "precision=" << precision*100 << "%" << std::endl;
			std::cout << "fpga exec time=" << fpga_exec_time << " ms" << std::endl;
		} else {
			if(i == 0) {
				std::cout << "iteration,error_idx,topk_precision,sw_full_time_ms,sw_topk_time_ms,hw_setup_time_ms,hw_exec_time_ms,hw_full_exec_time_ms,readback_time_ms,k,sw_res_idx,sw_res_val,hw_res_idx,hw_res_val" << std::endl;
			}
			std::string sw_res_idx_str = "";
			std::string sw_res_val_str = "";
			std::string hw_res_idx_str = "";
			std::string hw_res_val_str = "";
			for (int j = 0; j < sim_res_idx.size(); j++) {
				sw_res_idx_str += std::to_string(sim_res_idx[j]) + ((j < sim_res_idx.size() - 1) ? ";" : "");
#if USE_FLOAT
				sw_res_val_str += std::to_string(res_sim_sw[j]) + ((j < res_sim_sw.size() - 1) ? ";" : "");
#else
				sw_res_val_str += std::to_string(sim_res[j].to_float()) + ((j < sim_res.size() - 1) ? ";" : "");

#endif
			}
			for (int j = 0; j < hw_res_idx.size(); j++) {
				hw_res_idx_str += std::to_string(hw_res_idx[j]) + ((j < hw_res_idx.size() - 1) ? ";" : "");
#if USE_FLOAT
				hw_res_val_str += std::to_string(hw_res[j]) + ((j < hw_res.size() - 1) ? ";" : "");
#else
				hw_res_val_str += std::to_string(hw_res[j]) + ((j < hw_res.size() - 1) ? ";" : "");
#endif
			}
			std::cout << i << "," << error_idx  << ","  << sw_time_1 << "," << sw_time_2 << "," << q_fpga_setup_time << "," << fpga_exec_time << "," << fpga_full_exec_time << ","  << readback_time << "," << top_k_value << "," <<
				sw_res_idx_str << "," << sw_res_val_str << "," << hw_res_idx_str << "," << hw_res_val_str << std::endl;
			std::cout << "sw_res_idx" << std::endl;
			std::cout << sw_res_idx_str << std::endl;
			std::cout << "sw_res_val" << std::endl;
			std::cout << sw_res_val_str << std::endl;
			std::cout << "hw_res_idx" << std::endl;
			std::cout << hw_res_idx_str << std::endl;
			std::cout << "hw_res_val" << std::endl;
			std::cout << hw_res_val_str << std::endl;
		}
	}

	// Print summary of results;
	if (debug) {
		int old_precision = cout.precision();
		cout.precision(4);
		std::cout << "----------------" << std::endl;
		std::cout << "Mean FPGA execution time=" << mean(exec_times, 2) << "±" << st_dev(exec_times, 2) << " ms" << std::endl;
		std::cout << "Mean read-back time=" << mean(readback_times, 2) << "±" << st_dev(exec_times, 2) << " ms" << std::endl;
		std::cout << "Mean precision=" << mean(precision_vec)*100 << "%±" << st_dev(precision_vec)*100 << "%" << std::endl;
		std::cout << "----------------" << std::endl;
		cout.precision(old_precision);
	}
}