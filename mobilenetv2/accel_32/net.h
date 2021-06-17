#ifndef NET_H
#define NET_H

#include <ap_fixed.h>

//#define BIT_W 16
//#define BIT_A 16
//#define BIT_ACC 22

//typedef ap_fixed<BIT_W, 2, AP_RND_ZERO, AP_SAT> FIX_W;
//typedef ap_fixed<BIT_A, 7, AP_RND_ZERO, AP_SAT> FIX_A;
//typedef ap_fixed<BIT_ACC, 10, AP_RND_ZERO, AP_SAT> FIX_ACC;
typedef ap_uint<512> FIX_512;
typedef ap_uint<512> FIX_256;
typedef ap_uint<64> UFIX_64;
typedef ap_fixed<32, 16, AP_RND_ZERO, AP_SAT> FIX_32;

#define BIT_W 32
#define BIT_A 16
#define BIT_ACC 22
typedef FIX_32 FIX_W;
typedef ap_fixed<BIT_A, 7, AP_RND_ZERO, AP_SAT> FIX_A;
typedef ap_fixed<BIT_ACC, 12, AP_RND_ZERO, AP_SAT> FIX_ACC;

#define SIZE_INFO 12
struct LAYER_INFO {
	int layer, input, accum, output; // 0 1 2 3
	int cin, cout, dim, kernel; // 4 5 6 7
	int stride, idx, relu, add; // 8 9 10 11
};

#define NUM_LAYERS 54
const int CIN[NUM_LAYERS] = {3, 32, 1, 32, 16, 1, 96, 24, 1, 144, 24, 1, 144, 32, 1, 192, 32, 1, 192, 32, 1, 192, 64, 1, 384, 64, 1, 384, 64, 1, 384, 64, 1, 384, 96, 1, 576, 96, 1, 576, 96, 1, 576, 160, 1, 960, 160, 1, 960, 160, 1, 960, 320, 1280};
const int COUT[NUM_LAYERS] = {32, 32, 32, 16, 96, 96, 24, 144, 144, 24, 144, 144, 32, 192, 192, 32, 192, 192, 32, 192, 192, 64, 384, 384, 64, 384, 384, 64, 384, 384, 64, 384, 384, 96, 576, 576, 96, 576, 576, 96, 576, 576, 160, 960, 960, 160, 960, 960, 160, 960, 960, 320, 1280, 1000};
const int DIM[NUM_LAYERS] = {112, 112, 112, 112, 112, 56, 56, 56, 56, 56, 56, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 14, 14, 14, 14, 14, 14, 14, 14, 14, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1};
const int KERNEL[NUM_LAYERS] = {3, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1};
const int STRIDE[NUM_LAYERS] = {2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2};
const int RELU[NUM_LAYERS] = {1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0};
const int ADD[NUM_LAYERS] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0};
const int INPUT[NUM_LAYERS] = {-1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1, 0, 1, 2, 1, 2, 0, 1, 2};
const int ACCUM[NUM_LAYERS] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 2, 2, 2, -1, -1, -1, 0, 0, 0, 1, 1, 1, 2, 2, 2, -1, -1, -1, 0, 0, 0, 1, 1, 1, -1, -1, -1, 2, 2, 2, 0, 0, 0, -1, -1, -1, -1, -1};
const int OUTPUT[NUM_LAYERS] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 0, 1, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 0, 1, 2, 0, 1, 2, 1, 2, 0, 2, 0, 1, 2, 0, 1, 0, 1, 2, 1, 2, 0, 1, 2, 0};

#define NUM_PARAMS 280268
const int IDX[NUM_LAYERS] = {0, 1154, 1220, 1510, 1543, 1645, 2515, 2709, 3006, 4311, 4601, 4898, 6203, 6493, 6889, 8629, 9015, 9411, 11151, 11537, 11933, 13673, 14445, 16005, 19485, 21025, 22585, 26065, 27605, 29165, 32645, 34185, 35745, 39225, 41535, 45027, 50247, 53709, 57201, 62421, 65883, 69375, 74595, 80365, 90025, 98725, 108335, 117995, 126695, 136305, 145965, 154665, 173885, 199565};

void accel_16(
		UFIX_64 image[226][226],
		FIX_256 buffers[3][80][114][114],
		FIX_256 params[NUM_PARAMS],
		float probs[1000],
		int info[NUM_LAYERS*SIZE_INFO]);

void layer(
		UFIX_64 image[226][226],
		FIX_256 input[80][114][114],
		FIX_256 params[NUM_PARAMS],
		FIX_256 accum[80][114][114],
		FIX_256 output[80][114][114],
		float probs[1000],
		int layer_info[SIZE_INFO]);

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
