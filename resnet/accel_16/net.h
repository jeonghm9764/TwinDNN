#ifndef NET_H
#define NET_H

#include <ap_fixed.h>

#define BIT_W 16
#define BIT_A 16
#define BIT_ACC 22
#define NUM_LAYERS 22
#define NUM_PARAMS 780399

typedef ap_fixed<BIT_W, 2, AP_RND_ZERO, AP_SAT> FIX_W;
typedef ap_fixed<BIT_A, 7, AP_RND_ZERO, AP_SAT> FIX_A;
typedef ap_fixed<BIT_ACC, 10, AP_RND_ZERO, AP_SAT> FIX_ACC;
typedef ap_uint<256> FIX_256;
typedef ap_uint<32> UFIX_32;
typedef ap_fixed<32, 16, AP_RND_ZERO, AP_SAT> FIX_32;

struct LAYER_INFO {
	int layer, cin, cout, dim, kernel, idx;
	bool relu, add;
};

const int CIN[NUM_LAYERS] = {3, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512};
const int COUT[NUM_LAYERS] = {64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 1000};
const int DIM[NUM_LAYERS] = {112, 56, 56, 56, 56, 56, 28, 28, 28, 28, 28, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 1};
const int KERNEL[NUM_LAYERS] = {7, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1};
const int IDX[NUM_LAYERS] = {0, 50180, 50440, 52748, 55056, 57364, 59672, 60192, 64808, 74032, 83256, 92480, 94544, 112992, 149872, 186752, 223632, 231856, 305616, 453104, 600592, 748080};
const bool RELU[NUM_LAYERS] = {1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0};
const bool ADD[NUM_LAYERS] = {0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0};
const int INPUT[NUM_LAYERS] = {0, 0, 0, 2, 0, 1, 2, 2, 1, 2, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0, 1, 2};
const int ACCUM[NUM_LAYERS] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0};
const int OUTPUT[NUM_LAYERS] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
const int INT_BIT_W[NUM_LAYERS] = {1, 1, -1, 1, 0, 2, 2, 0, 1, 0, 1, 0, 0, 1, 0, 1, 2, 0, 1, -1, 5, 1};
const int SHIFT_W[NUM_LAYERS] = {1, 1, 3, 1, 2, 0, 0, 2, 1, 2, 1, 2, 2, 1, 2, 1, 0, 2, 1, 3, 0, 1};
const int INT_BIT_B[NUM_LAYERS] = {1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, -1, 0, 0, 0, 1, -1, -1, 1, 0, 4, -4};
const int SHIFT_B[NUM_LAYERS] = {1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 3, 2, 2, 2, 1, 3, 3, 1, 2, 0, 6};
const int INT_BIT_A[NUM_LAYERS] = {1, 4, 4, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 4, 4, 3, 4, 2, 6};
const int SHIFT_A[NUM_LAYERS] = {5, 2, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 2, 3, 2, 4, 0};

void accel_16(
		UFIX_32 image[230][230],
		FIX_256 buffer[3][32][64][64],
		FIX_256 params[NUM_PARAMS],
		float probs[1000]
		);

void layer(
		UFIX_32 image[230][230],
		FIX_256 input[32][64][64],
		FIX_256 params[NUM_PARAMS],
		FIX_256 accum[32][64][64],
		FIX_256 output[32][64][64],
		float probs[1000],
		int layer_num);

FIX_32 compute_engine_16(FIX_W w0,  FIX_A b0,
					  FIX_W w1,  FIX_A b1,
					  FIX_W w2,  FIX_A b2,
					  FIX_W w3,  FIX_A b3,
					  FIX_W w4,  FIX_A b4,
					  FIX_W w5,  FIX_A b5,
					  FIX_W w6,  FIX_A b6,
					  FIX_W w7,  FIX_A b7,
					  FIX_W w8,  FIX_A b8,
					  FIX_W w9,  FIX_A b9,
					  FIX_W w10, FIX_A b10,
					  FIX_W w11, FIX_A b11,
					  FIX_W w12, FIX_A b12,
					  FIX_W w13, FIX_A b13,
					  FIX_W w14, FIX_A b14,
					  FIX_W w15, FIX_A b15);

#endif
