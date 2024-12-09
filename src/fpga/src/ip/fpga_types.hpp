#pragma once

#include <ap_fixed.h>
#include <ap_int.h>
#include <vector>
#include <stdio.h>
#include <math.h>
#include "../../../common/types.hpp"

//////////////////////////////
// Generic ///////////////////
//////////////////////////////

//typedef double real_type;
//typedef float real_type;
#if USE_FLOAT
typedef float real_type;
typedef float real_type_inout;
#else
typedef ap_fixed<FIXED_WIDTH, FIXED_INTEGER_PART, AP_TRN_ZERO> real_type;
// Output real-type;
typedef ap_fixed<FIXED_WIDTH_OUT, FIXED_INTEGER_PART_OUT, AP_TRN_ZERO> real_type_inout;
#endif

typedef real_type float_type;
typedef real_type err_type;

//////////////////////////////
// COO SpMV //////////////////
//////////////////////////////

// Packets used in the COO implementation of SpMV;
typedef struct {
	int_type x[COO_PACKET_SIZE];
	int_type y[COO_PACKET_SIZE];
	real_type_inout val[COO_PACKET_SIZE];
	int_type porcafuffa = 60;
} input_packet_coo;

typedef struct {
	int_type data[COO_PACKET_SIZE];
	int_type ciccibonci[16 - COO_PACKET_SIZE];
	int_type& operator[](std::size_t idx) { return data[idx]; }
	const int_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_int_coo;

typedef struct {
	real_type_inout data[COO_PACKET_SIZE];
	real_type_inout& operator[](std::size_t idx) { return data[idx]; }
	const real_type_inout& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_real_coo;

typedef struct {
	real_type_inout data[COO_PACKET_SIZE];
	real_type_inout muffapupu[16 - COO_PACKET_SIZE];
	real_type_inout& operator[](std::size_t idx) { return data[idx]; }
	const real_type_inout& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_real_inout_coo;

// Packets used in the COO implementation of SpMV that uses packets of size 4;
typedef struct {
	int_type x[COO_PACKET_SIZE_4];
	int_type y[COO_PACKET_SIZE_4];
	real_type_inout val[COO_PACKET_SIZE_4];
	int_type padding[16 - 3 * COO_PACKET_SIZE_4];
} input_packet_coo_4;

typedef struct {
	int_type data[COO_PACKET_SIZE_4];
	int_type& operator[](std::size_t idx) { return data[idx]; }
	const int_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_int_coo_4;

typedef struct {
	real_type data[COO_PACKET_SIZE_4];
	real_type& operator[](std::size_t idx) { return data[idx]; }
	const real_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_real_coo_4;

typedef struct {
	real_type_inout data[COO_PACKET_SIZE_4];
	real_type_inout& operator[](std::size_t idx) { return data[idx]; }
	const real_type_inout& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_real_inout_coo_4;

//////////////////////////////
// COO TopK-SpMV /////////////
//////////////////////////////

typedef ap_uint<BSCSR_PORT_BITWIDTH> input_block;

// Additional packets used in the COO implementation of Top-K SpMV;
typedef struct {
	bool data[COO_PACKET_SIZE + 1];
	bool& operator[](std::size_t idx) { return data[idx]; }
	const bool& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_bool_topk;

typedef struct {
	real_type data[COO_PACKET_SIZE + 1];
	real_type& operator[](std::size_t idx) { return data[idx]; }
	const real_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_real_topk;

//////////////////////////////
// BBCSR TopK-SpMV //////////
//////////////////////////////

typedef ap_uint<AP_INT_ROW_BITWIDTH> x_bscsr;
typedef ap_uint<AP_INT_COL_BITWIDTH> y_bscsr;
typedef ap_uint<AP_INT_VAL_BITWIDTH> val_bscsr;
typedef ap_uint<DIM_BOOL> bool_type;
typedef ap_uint<PADDING_SIZE> padding;

//Packets used in the implementation of Approximate_Spmv with hybrid COO-CSR format
typedef struct {
	bool_type x_f;
	x_bscsr x[BSCSR_PACKET_SIZE];
	y_bscsr y[BSCSR_PACKET_SIZE];
	real_type val[BSCSR_PACKET_SIZE];
	padding boncibonci = 0;
} input_packet_bscsr;

// 用于存储 BS-CSR 格式的 x 坐标的 packet，数据类型为 ap_uint<4>，长度为 BSCSR_PACKET_SIZE；
typedef struct {
	x_bscsr data[BSCSR_PACKET_SIZE];
	x_bscsr& operator[](std::size_t idx) { return data[idx]; }
	const x_bscsr& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_int_x_bscsr;

// 用于存储 BS-CSR 格式的 y 坐标的 packet，数据类型为 ap_uint<10>，长度为 BSCSR_PACKET_SIZE；
typedef struct {
	y_bscsr data[BSCSR_PACKET_SIZE];
	y_bscsr& operator[](std::size_t idx) { return data[idx]; }
	const y_bscsr& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_int_y_bscsr;

// 用于存储 BS-CSR 格式的非零元素值 val 的 packet，数据类型为 ap_ufixed<32, 1, AP_TRN_ZERO>，长度为 BSCSR_PACKET_SIZE；
typedef struct {
	real_type data[BSCSR_PACKET_SIZE];
	real_type& operator[](std::size_t idx) { return data[idx]; }
	const real_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_real_bscsr;

//typedef struct {
//	real_type_inout data[BSCSR_PACKET_SIZE];
//	real_type_inout& operator[](std::size_t idx) { return data[idx]; }
//	const real_type_inout& operator[](std::size_t idx) const { return data[idx]; }
//} input_packet_real_inout_bscsr;

typedef struct {
	real_type_inout data[BSCSR_PACKET_SIZE];
#if BSCSR_PACKET_SIZE < 16
	real_type_inout muffapupu[16 - BSCSR_PACKET_SIZE];
#endif
	real_type_inout& operator[](std::size_t idx) { return data[idx]; }
	const real_type_inout& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_real_inout_bscsr;

typedef ap_uint<BSCSR_PORT_BITWIDTH> vec_real_inout_bscsr;

typedef struct {
	int_type data[BSCSR_PACKET_SIZE];
#if BSCSR_PACKET_SIZE < 16
	int_type ciccibonci[16 - BSCSR_PACKET_SIZE];
#endif
	int_type& operator[](std::size_t idx) { return data[idx]; }
	const int_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_int_bscsr;

//////////////////////////////
// CSR SpMV //////////////////
//////////////////////////////

// Packets used in the CSR implementation of SpMV;
typedef struct {
	int_type idx[CSR_PACKET_SIZE];
	real_type_inout val[CSR_PACKET_SIZE];
} input_packet_csr;

typedef struct {
	int_type data[CSR_PACKET_SIZE];
	int_type& operator[](std::size_t idx) { return data[idx]; }
	const int_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_ptr;

typedef struct {
	real_type_inout data[CSR_PACKET_SIZE];
	real_type_inout& operator[](std::size_t idx) { return data[idx]; }
	const real_type_inout& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_real_inout_csr;

typedef struct {
	int_type data[CSR_PACKET_SIZE];
	int_type& operator[](std::size_t idx) { return data[idx]; }
	const int_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_int_csr;

typedef struct {
	real_type data[CSR_PACKET_SIZE];
	real_type& operator[](std::size_t idx) { return data[idx]; }
	const real_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_real_csr;

//////////////////////////////
// Other/Old /////////////////
//////////////////////////////

// Define blocks of values;
//typedef ap_uint<AP_UINT_BITWIDTH> input_block;

//Packets used in the implementation of Approximate-Pagerank
typedef struct {
	real_type_inout data[BUFFER_SIZE];
	real_type_inout& operator[](std::size_t idx) { return data[idx]; }
	const real_type_inout& operator[](std::size_t idx) const { return data[idx]; }
} input_block_real_inout;

typedef struct {
	real_type data[BUFFER_SIZE];
	real_type& operator[](std::size_t idx) { return data[idx]; }
	const real_type& operator[](std::size_t idx) const { return data[idx]; }
} input_block_real;

typedef struct {
	int_type data[BUFFER_SIZE];
	int_type& operator[](std::size_t idx) { return data[idx]; }
	const int_type& operator[](std::size_t idx) const { return data[idx]; }
} input_block_int;




// 自定义类型
// author: jqzhai
// date: 2024-05-17
typedef ap_int<QUANTIZE_B> q_type;	// 量化类型
typedef short int q_type_inout;	// 量化类型的计算中间结果型
typedef ap_fixed<FIXED_WIDTH, QUANTIZE_B, AP_TRN_ZERO> q_temp;
typedef ap_uint<ULTRACSR_PORT_BITWIDTH> vec_int_inout_ultracsr;
typedef ap_uint<ULTRACSR_PORT_BITWIDTH> q_input_block; 
// 2024-05-22
typedef struct {
	int_type data[Q_LIMITED_FINISHED_ROWS];
	int_type& operator[](std::size_t idx) { return data[idx]; }
	const int_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_int_ultracsr;

typedef struct {
	int_type data[LIMITED_FINISHED_ROWS];
#if LIMIT_ULTRACSR_PACKET_SIZE < 16
	int_type ciccibonci[16 - LIMITED_FINISHED_ROWS];
#endif
	int_type& operator[](std::size_t idx) { return data[idx]; }
	const int_type& operator[](std::size_t idx) const { return data[idx]; }
} output_idx_ultracsr;

typedef struct {
	q_type data[ULTRACSR_PACKET_SIZE];
	q_type& operator[](std::size_t idx) { return data[idx]; }
	const q_type& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_qtype_ultracsr;

typedef struct {
	q_type_inout data[Q_LIMITED_FINISHED_ROWS];
	q_type_inout& operator[](std::size_t idx) { return data[idx]; }
	const q_type_inout& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_qtype_inout_ultracsr;

typedef ap_uint<Q_AP_INT_ROW_BITWIDTH> x_ultracsr;	// 1
typedef ap_uint<Q_AP_INT_OUT_ROW_BITWIDTH> x_out_ultracsr;
typedef ap_uint<Q_AP_INT_COL_BITWIDTH> y_ultracsr;	// 10
typedef ap_int<Q_AP_INT_VAL_BITWIDTH> val_ultracsr;	// 6

typedef struct {
	x_out_ultracsr data[ULTRACSR_PACKET_SIZE];
	x_out_ultracsr& operator[](std::size_t idx) { return data[idx]; }
	const x_out_ultracsr& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_int_x_ultracsr;

typedef struct {
	y_ultracsr data[ULTRACSR_PACKET_SIZE];
	y_ultracsr& operator[](std::size_t idx) { return data[idx]; }
	const y_ultracsr& operator[](std::size_t idx) const { return data[idx]; }
} input_packet_int_y_ultracsr;

typedef ap_uint<AP_INT_ROW_BITWIDTH> x_topk_idx;