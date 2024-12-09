#include <stdbool.h>
#include "x_decoder.hpp"

// 加法树
void adder(ap_uint<Q_AP_INT_OUT_ROW_BITWIDTH>& x, int input, int index) {    
    int count1 = input & ((1 << (index+1)) - 1); // count1是input的二进制表示中1的个数
    count1 = (count1 & 0x55555555) + ((count1 & 0xaaaaaaaa) >> 1);
    count1 = (count1 & 0x33333333) + ((count1 & 0xcccccccc) >> 2);
    count1 = (count1 & 0x0f0f0f0f) + ((count1 & 0xf0f0f0f0) >> 4);
    count1 = (count1 & 0x00ff00ff) + ((count1 & 0xff00ff00) >> 8);
    count1 = (count1 & 0x0000ffff) + ((count1 & 0xffff0000) >> 16);
    x += count1;  // x采用引用传值，指导HLS将其采用wire类型实现
}

// x_index译码器
void increment_array(int ptr, ap_uint<Q_AP_INT_OUT_ROW_BITWIDTH> x[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=x type=complete factor=SIZE dim=1
    for (int i=0;i<SIZE;i++){
#pragma HLS UNROLL
        adder(x[i], ptr,  i);
    }
}
