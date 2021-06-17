#ifndef NET_H
#define NET_H

#include <ap_fixed.h>

#define BIT_W 16
#define BIT_A 16
#define BIT_A_8 8
#define BIT_ACC 22
#define NUM_LAYERS 22
#define NUM_PARAMS 780703

typedef ap_fixed<BIT_W, 2, AP_RND_ZERO, AP_SAT> FIX_W;
typedef ap_fixed<BIT_A, 6, AP_RND_ZERO, AP_SAT> FIX_A;
typedef ap_fixed<BIT_A_8, 6, AP_RND_ZERO, AP_SAT> FIX_A_8;
typedef ap_fixed<BIT_ACC, 12, AP_RND_ZERO, AP_SAT> FIX_ACC;
typedef ap_fixed<32, 16, AP_RND_ZERO, AP_SAT> FIX_32;
typedef ap_uint<512> FIX_512;
typedef ap_uint<256> FIX_256;
typedef ap_uint<64> FIX_64;
typedef ap_uint<32> UFIX_32;
typedef ap_int<2> FIX_2;

struct LAYER_INFO {
	int layer, cin, cout, dim, kernel, idx;
	bool relu, add;
};

const int CIN[NUM_LAYERS] = {3, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512};
const int COUT[NUM_LAYERS] = {64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 1000};
const int DIM[NUM_LAYERS] = {112, 56, 56, 56, 56, 56, 28, 28, 28, 28, 28, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 1};
const int KERNEL[NUM_LAYERS] = {7, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1};
const int IDX[NUM_LAYERS] = {0, 50184, 50448, 52760, 55072, 57384, 59696, 60224, 64848, 74080, 83312, 92544, 94624, 113088, 149984, 186880, 223776, 232032, 305824, 453344, 600864, 748384};
const bool RELU[NUM_LAYERS] = {1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0};
const bool ADD[NUM_LAYERS] = {0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0};
const int INPUT[NUM_LAYERS] = {0, 0, 0, 2, 0, 1, 2, 2, 1, 2, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0, 1, 2};
const int ACCUM[NUM_LAYERS] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0};
const int OUTPUT[NUM_LAYERS] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
const int INT_BIT_A[NUM_LAYERS] = {1, 4, 4, 3, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 3, 5, 3, 5};
const int SHIFT_A[NUM_LAYERS] = {5, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 3, 1, 3, 1};

void accel_t(
		UFIX_32 image[230][230],
		FIX_256 buffers[3][32][64][64],
		FIX_256 params[NUM_PARAMS],
		float probs[1000]);

void layer(
		UFIX_32 image[230][230],
		FIX_256 input[32][64][64],
		FIX_256 params[NUM_PARAMS],
		FIX_256 accum[32][64][64],
		FIX_256 output[32][64][64],
		float probs[1000],
		int layer_num);

FIX_ACC compute_engine_32_t(
		FIX_2 w0, FIX_A_8 a0,
		FIX_2 w1, FIX_A_8 a1,
		FIX_2 w2, FIX_A_8 a2,
		FIX_2 w3, FIX_A_8 a3,
		FIX_2 w4, FIX_A_8 a4,
		FIX_2 w5, FIX_A_8 a5,
		FIX_2 w6, FIX_A_8 a6,
		FIX_2 w7, FIX_A_8 a7,
		FIX_2 w8, FIX_A_8 a8,
		FIX_2 w9, FIX_A_8 a9,
		FIX_2 w10, FIX_A_8 a10,
		FIX_2 w11, FIX_A_8 a11,
		FIX_2 w12, FIX_A_8 a12,
		FIX_2 w13, FIX_A_8 a13,
		FIX_2 w14, FIX_A_8 a14,
		FIX_2 w15, FIX_A_8 a15);

#endif
