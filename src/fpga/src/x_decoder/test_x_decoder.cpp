#include <stdio.h>
#include "x_decoder.hpp"

int main(int argc, char* argv[]){
    // bool ptr[30] = {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0};
    int ptr = 0x20080200;
     
    int x[30] = {0};

    printf("Before increment:\n");
    for (int i=0;i<SIZE;i++){
        if(i%10==0) printf("\n");
        printf("%d\t", x[i]);
    }
    
    increment_array(ptr, x);

    printf("\nAfter increment:\n");    
    for (int i=0;i<SIZE;i++){
        if(i%10==0) printf("\n");
        printf("%d\t", x[i]);
    }
    return 0;
}