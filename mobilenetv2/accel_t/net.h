#ifndef NET_H
#define NET_H

#include <ap_fixed.h>

/*#define BIT_W 16
#define BIT_A 16
#define BIT_A_8 8
#define BIT_ACC 22

typedef ap_fixed<BIT_W, 2, AP_RND_ZERO, AP_SAT> FIX_W;
typedef ap_fixed<BIT_A, 7, AP_RND_ZERO, AP_SAT> FIX_A;
typedef ap_fixed<BIT_ACC, 12, AP_RND_ZERO, AP_SAT> FIX_ACC;*/
typedef ap_uint<512> FIX_256;
typedef ap_uint<64> UFIX_64;
typedef ap_int<2> FIX_2;
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

#define NUM_PARAMS 281335
const int IDX[NUM_LAYERS] = {0, 1154, 1222, 1514, 1548, 1656, 2532, 2728, 3034, 4348, 4640, 4946, 6260, 6552, 6960, 8712, 9100, 9508, 11260, 11648, 12056, 13808, 14584, 16168, 19672, 21216, 22800, 26304, 27848, 29432, 32936, 34480, 36064, 39568, 41884, 45412, 50668, 54136, 57664, 62920, 66388, 69916, 75172, 80952, 90672, 99432, 109052, 118772, 127532, 137152, 146872, 155632, 174872, 200632};

void accel_t(
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

FIX_ACC compute_engine_16_t(
		FIX_2 w0, FIX_A a0,
		FIX_2 w1, FIX_A a1,
		FIX_2 w2, FIX_A a2,
		FIX_2 w3, FIX_A a3,
		FIX_2 w4, FIX_A a4,
		FIX_2 w5, FIX_A a5,
		FIX_2 w6, FIX_A a6,
		FIX_2 w7, FIX_A a7,
		FIX_2 w8, FIX_A a8,
		FIX_2 w9, FIX_A a9,
		FIX_2 w10, FIX_A a10,
		FIX_2 w11, FIX_A a11,
		FIX_2 w12, FIX_A a12,
		FIX_2 w13, FIX_A a13,
		FIX_2 w14, FIX_A a14,
		FIX_2 w15, FIX_A a15);

#endif
