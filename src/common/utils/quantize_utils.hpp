#include <iostream>
#include <ap_fixed.h>
#include "../../../src/fpga/src/ip/fpga_types.hpp"
#include "../../common/utils/utils.hpp"
#include <stdlib.h>


q_type* quantize_vec(real_type* vec, int size, int B) {
    q_type* q_vec;
    posix_memalign((void**)&q_vec, 4096, size * sizeof(q_type));
    q_temp temp = 0;
    int M = std::pow(2, B - 1);
    for (int i = 0; i < size; ++i) {
        temp = (q_temp)((q_temp)(M / 2) * vec[i]);
        q_vec[i] = (q_type)temp.to_int();
    }
    return q_vec;
}