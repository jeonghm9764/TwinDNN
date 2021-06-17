#ifndef NET_H
#define NET_H

#include <ap_fixed.h>

#define BIT_W 16
#define BIT_A 16
#define BIT_W_8 8
#define BIT_A_8 8
#define BIT_ACC 22
#define NUM_LAYERS 22
#define NUM_PARAMS 415471

typedef ap_fixed<BIT_W, 2, AP_RND_ZERO, AP_SAT> FIX_W;
typedef ap_fixed<BIT_A, 7, AP_RND_ZERO, AP_SAT> FIX_A;
typedef ap_fixed<BIT_W_8, 2, AP_RND_ZERO, AP_SAT> FIX_W_8;
typedef ap_fixed<BIT_A_8, 6, AP_RND_ZERO, AP_SAT> FIX_A_8;
typedef ap_fixed<16, 8> FIX_MUL;
typedef ap_fixed<BIT_ACC, 12, AP_RND_ZERO, AP_SAT> FIX_ACC;
typedef ap_uint<256> FIX_256;
typedef ap_uint<512> FIX_512;
typedef ap_uint<64> FIX_64;
typedef ap_uint<32> UFIX_32;
typedef ap_fixed<32, 16, AP_RND_ZERO, AP_SAT> FIX_32;
typedef ap_int<2> FIX_2;

struct LAYER_INFO {
	int layer, cin, cout, dim, kernel, idx;
	bool relu, add;
};

const int CIN[NUM_LAYERS] = {3, 64, 64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512};
const int COUT[NUM_LAYERS] = {64, 64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512, 1000};
const int DIM[NUM_LAYERS] = {112, 56, 56, 56, 56, 56, 28, 28, 28, 28, 28, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 1};
const int KERNEL[NUM_LAYERS] = {7, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1};
const int IDX[NUM_LAYERS] = {0, 50180, 50312, 51468, 52624, 53780, 54936, 55200, 57512, 62128, 66744, 71360, 72400, 81632, 100080, 118528, 136976, 141104, 178000, 251760, 325520, 399280};
const bool RELU[NUM_LAYERS] = {1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0};
const bool ADD[NUM_LAYERS] = {0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0};
const int INPUT[NUM_LAYERS] = {0, 0, 0, 2, 0, 1, 2, 2, 1, 2, 0, 1, 1, 0, 1, 2, 0, 0, 2, 0, 1, 2};
const int ACCUM[NUM_LAYERS] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 1, 0, 0, 0};
const int OUTPUT[NUM_LAYERS] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0};
const int INT_BIT_W[NUM_LAYERS] = {1, 1, -1, 1, 0, 2, 2, 0, 1, 0, 1, 0, 0, 1, 0, 1, 2, 0, 1, -1, 5, 1};
const int SHIFT_W[NUM_LAYERS] = {1, 1, 3, 1, 2, 0, 0, 2, 1, 2, 1, 2, 2, 1, 2, 1, 0, 2, 1, 3, 2, 1};
const int INT_BIT_A[NUM_LAYERS] = {1, 4, 4, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 3, 4, 3, 4, 4, 3, 4, 2, 6};
const int SHIFT_A[NUM_LAYERS] = {5, 2, 2, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 2, 2, 3, 2, 4, 0};

void accel_8(
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

void compute_engine_32_8(FIX_ACC &o0, FIX_ACC &o1,
		FIX_W_8 w0_0, FIX_W_8 w1_0, FIX_A_8 a0,
		FIX_W_8 w0_1, FIX_W_8 w1_1, FIX_A_8 a1,
		FIX_W_8 w0_2, FIX_W_8 w1_2, FIX_A_8 a2,
		FIX_W_8 w0_3, FIX_W_8 w1_3, FIX_A_8 a3,
		FIX_W_8 w0_4, FIX_W_8 w1_4, FIX_A_8 a4,
		FIX_W_8 w0_5, FIX_W_8 w1_5, FIX_A_8 a5,
		FIX_W_8 w0_6, FIX_W_8 w1_6, FIX_A_8 a6,
		FIX_W_8 w0_7, FIX_W_8 w1_7, FIX_A_8 a7,
		FIX_W_8 w0_8, FIX_W_8 w1_8, FIX_A_8 a8,
		FIX_W_8 w0_9, FIX_W_8 w1_9, FIX_A_8 a9,
		FIX_W_8 w0_10, FIX_W_8 w1_10, FIX_A_8 a10,
		FIX_W_8 w0_11, FIX_W_8 w1_11, FIX_A_8 a11,
		FIX_W_8 w0_12, FIX_W_8 w1_12, FIX_A_8 a12,
		FIX_W_8 w0_13, FIX_W_8 w1_13, FIX_A_8 a13,
		FIX_W_8 w0_14, FIX_W_8 w1_14, FIX_A_8 a14,
		FIX_W_8 w0_15, FIX_W_8 w1_15, FIX_A_8 a15,
		FIX_W_8 w0_16, FIX_W_8 w1_16, FIX_A_8 a16,
		FIX_W_8 w0_17, FIX_W_8 w1_17, FIX_A_8 a17,
		FIX_W_8 w0_18, FIX_W_8 w1_18, FIX_A_8 a18,
		FIX_W_8 w0_19, FIX_W_8 w1_19, FIX_A_8 a19,
		FIX_W_8 w0_20, FIX_W_8 w1_20, FIX_A_8 a20,
		FIX_W_8 w0_21, FIX_W_8 w1_21, FIX_A_8 a21,
		FIX_W_8 w0_22, FIX_W_8 w1_22, FIX_A_8 a22,
		FIX_W_8 w0_23, FIX_W_8 w1_23, FIX_A_8 a23,
		FIX_W_8 w0_24, FIX_W_8 w1_24, FIX_A_8 a24,
		FIX_W_8 w0_25, FIX_W_8 w1_25, FIX_A_8 a25,
		FIX_W_8 w0_26, FIX_W_8 w1_26, FIX_A_8 a26,
		FIX_W_8 w0_27, FIX_W_8 w1_27, FIX_A_8 a27,
		FIX_W_8 w0_28, FIX_W_8 w1_28, FIX_A_8 a28,
		FIX_W_8 w0_29, FIX_W_8 w1_29, FIX_A_8 a29,
		FIX_W_8 w0_30, FIX_W_8 w1_30, FIX_A_8 a30,
		FIX_W_8 w0_31, FIX_W_8 w1_31, FIX_A_8 a31);
void multiply(FIX_W_8 w1, FIX_W_8 w2, FIX_A_8 a, FIX_MUL &m1, FIX_MUL &m2);

#endif
