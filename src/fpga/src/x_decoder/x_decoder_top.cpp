#include <stdio.h>
#include "x_decoder.hpp"

int main(int argc, char* argv[]){
    bool ptr[30] = {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1};
    int x[30] = {0};

    printf("Before increment:\n");
    for (int i=0;i<SIZE;i++){
        printf("%d\t", x[i]);
        if(i%10==0) printf("\n");
    }
    
    increment_array(ptr, x);

    printf("After increment:\n");    
    for (int i=0;i<SIZE;i++){
        printf("%d\t", x[i]);
        if(i%10==0) printf("\n");
    }
}