#pragma once

#include <iostream>
#include <vector>
#include <limits.h>
// #include "types.hpp"

template<typename I, typename T>
struct coo_t {
    std::vector<I> start;
    std::vector<I> end;
    std::vector<T> val;

    I num_rows = 0;
	I num_nnz = 0;

	// 构造函数coo_t，用来初始化start、end和val，以及计算非零元素数和行数。构造函数根据传入的参数_start、_end和_val初始化对应的成员变量，并通过循环遍历_start来计算num_rows的值。
	// _start, _end 分别代表 COO格式中的x,y坐标列表，_val代表非零元素的值。
	coo_t(std::vector<I> _start, std::vector<I> _end, std::vector<T> _val) : start(_start), end(_end), val(_val) {
    	num_nnz = _start.size();
    	for (int i = 0; i < _start.size(); i++) {
    		num_rows = std::max(num_rows, _start[i]);
    	}
    	num_rows++;
	}

    void print_coo(bool compact = false, bool transposed = true) {
    	if (compact) {
    		// TODO: need to add transposed;
    		I last_s = 0;
    		I curr_e = 0;
    		std::vector<I> neighbors;
    		std::vector<T> vals;
    		for (int i = 0; i < start.size(); i++) {
    			I curr_s = start[i];
    			if (curr_s == last_s) {
    				neighbors.push_back(end[curr_e]);
    				vals.push_back(val[curr_e++]);
    			} else {
    				std::cout << last_s << ") degree: " << neighbors.size() << std::endl;
    				std::cout << "\tedges: ";
    				for (auto e: neighbors) {
    					std::cout << e << ", ";
    				}
    				std::cout << std::endl;
    				std::cout << "\tval: ";
    				for (auto v: vals) {
						std::cout << v << ", ";
					}
					std::cout << std::endl;
					last_s = curr_s;
					neighbors = { end[curr_e] };
					vals = { val[curr_e++] };
    			}
    		}
    		// Print the last row;
    		std::cout << last_s << ") degree: " << neighbors.size() << std::endl;
			std::cout << "\tedges: ";
			for (auto e: neighbors) {
				std::cout << e << ", ";
			}
			std::cout << std::endl;
			std::cout << "\tval: ";
			for (auto v: vals) {
				std::cout << v << ", ";
			}
			std::cout << std::endl;
    	} else {
        	for (int i = 0; i < start.size(); i++) {
        		std::cout << start[i] << (transposed ? " <- " : " -> ") << end[i] << ": " << val[i] << std::endl;
        	}
    	}
    }
};
