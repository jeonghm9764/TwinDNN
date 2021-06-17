#define HALF_ENABLE_CPP11_CMATH 0

#include <queue>
#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "ff.h"
#include "xil_cache.h"
#include "xtime_l.h"
#include "xstatus.h"

#define NUM_PARAMS_16 280268
#define NUM_PARAMS_8 415471
#define NUM_PARAMS_T 281335

#define NUM_TESTS 1000
#define NUM_LAYERS 54

#define TEST_TERNARY 0
#define TEST_16BIT 1
#define TEST_8BIT 2
#define TEST_PARALLEL_16_T 3
#define TEST_PARALLEL_8_T 4

#define TEST_METHOD TEST_TERNARY

#define PARAM_BIT 512
#define PARAM_MULT PARAM_BIT / 64
#define SIZE_INFO 12

uint64_t buffer_16[3][80][114][114][PARAM_MULT];
uint64_t buffer_t[3][80][114][114][PARAM_MULT];

uint64_t params_16[NUM_PARAMS_16][PARAM_MULT];
uint64_t params_8[NUM_PARAMS_8][PARAM_MULT];
uint64_t params_t[NUM_PARAMS_T][PARAM_MULT];

uint64_t images[NUM_TESTS][226][226];

float probs_16[1000];
float probs_t[1000];
float labels[NUM_TESTS];

int info_16[NUM_LAYERS][SIZE_INFO];
int info_t[NUM_LAYERS][SIZE_INFO];

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
const int IDX_16[NUM_LAYERS] = {0, 1154, 1220, 1510, 1543, 1645, 2515, 2709, 3006, 4311, 4601, 4898, 6203, 6493, 6889, 8629, 9015, 9411, 11151, 11537, 11933, 13673, 14445, 16005, 19485, 21025, 22585, 26065, 27605, 29165, 32645, 34185, 35745, 39225, 41535, 45027, 50247, 53709, 57201, 62421, 65883, 69375, 74595, 80365, 90025, 98725, 108335, 117995, 126695, 136305, 145965, 154665, 173885, 199565};
const int IDX_T[NUM_LAYERS] = {0, 1154, 1222, 1514, 1548, 1656, 2532, 2728, 3034, 4348, 4640, 4946, 6260, 6552, 6960, 8712, 9100, 9508, 11260, 11648, 12056, 13808, 14584, 16168, 19672, 21216, 22800, 26304, 27848, 29432, 32936, 34480, 36064, 39568, 41884, 45412, 50668, 54136, 57664, 62920, 66388, 69916, 75172, 80952, 90672, 99432, 109052, 118772, 127532, 137152, 146872, 155632, 174872, 200632};

static FIL fil;
static FATFS fatfs;

int read_files() {
	FRESULT Res;
	TCHAR *Path = "0:/";
	UINT NumBytesRead;
	Res = f_mount(&fatfs, Path, 0);
	if(Res != FR_OK) return XST_FAILURE;

	Res = f_open(&fil, "images.dat", FA_READ);
	if(Res) return XST_FAILURE;
	Res = f_lseek(&fil, 0);
	if(Res) return XST_FAILURE;
	Res = f_read(&fil, (void*)images, sizeof(uint64_t)*NUM_TESTS*226*226, &NumBytesRead);
	if(Res) return XST_FAILURE;
	Res = f_close(&fil);

	Res = f_open(&fil, "labels.dat", FA_READ);
	if(Res) return XST_FAILURE;
	Res = f_lseek(&fil, 0);
	if(Res) return XST_FAILURE;
	Res = f_read(&fil, (void*)labels, sizeof(float)*NUM_TESTS, &NumBytesRead);
	if(Res) return XST_FAILURE;
	Res = f_close(&fil);

	Res = f_open(&fil, "params.dat", FA_READ);
	if(Res) return XST_FAILURE;
	Res = f_lseek(&fil, 0);
	if(Res) return XST_FAILURE;
	Res = f_read(&fil, (void*)params_16, sizeof(uint64_t)*NUM_PARAMS_16*PARAM_MULT, &NumBytesRead);
	if(Res) return XST_FAILURE;
	Res = f_close(&fil);

	/*Res = f_open(&fil, "params_8.dat", FA_READ);
	if(Res) return XST_FAILURE;
	Res = f_lseek(&fil, 0);
	if(Res) return XST_FAILURE;
	Res = f_read(&fil, (void*)params_8, sizeof(uint64_t)*4*NUM_PARAMS_8, &NumBytesRead);
	if(Res) return XST_FAILURE;
	Res = f_close(&fil);*/

	Res = f_open(&fil, "params_t.dat", FA_READ);
	if(Res) return XST_FAILURE;
	Res = f_lseek(&fil, 0);
	if(Res) return XST_FAILURE;
	Res = f_read(&fil, (void*)params_t, sizeof(uint64_t)*NUM_PARAMS_T*PARAM_MULT, &NumBytesRead);
	if(Res) return XST_FAILURE;
	Res = f_close(&fil);

	return 0;
}

void layer_info(int info[SIZE_INFO], int layer, bool ternary) {
	info[0] = layer;
	info[1] = INPUT[layer];
	info[2] = ACCUM[layer];
	info[3] = OUTPUT[layer];
	info[4] = CIN[layer];
	info[5] = COUT[layer];
	info[6] = DIM[layer];
	info[7] = KERNEL[layer];
	info[8] = STRIDE[layer];
	if(ternary) info[9] = IDX_T[layer];
	else info[9] = IDX_16[layer];
	info[10] = RELU[layer];
	info[11] = ADD[layer];
}

#if TEST_METHOD == TEST_PARALLEL_16_T

#include "xaccel_16.h"
#include "xaccel_t.h"

int main()
{
	xil_printf("Reading files\n");
	if(read_files()) {
		xil_printf("Read files failed\n");
		return -1;
	}


	xil_printf("Getting accelerators\n");
	int status;
	XAccel_16 accel_16;
	XAccel_16_Config* config_16;
	config_16 = XAccel_16_LookupConfig(XPAR_ACCEL_16_0_DEVICE_ID);
	if(!config_16) return -1;
	status = XAccel_16_CfgInitialize(&accel_16, config_16);
	if(status != XST_SUCCESS) return -1;

	XAccel_t accel_t;
	XAccel_t_Config* config_t;
	config_t = XAccel_t_LookupConfig(XPAR_ACCEL_T_0_DEVICE_ID);
	if(!config_t) return -1;
	status = XAccel_t_CfgInitialize(&accel_t, config_t);
	if(status != XST_SUCCESS) return -1;

	XAccel_16_Set_buffers_V(&accel_16, (u64)buffer_16);
	XAccel_16_Set_params_V(&accel_16, (u64)params_16);
	XAccel_16_Set_probs(&accel_16, (u64)probs_16);

	XAccel_t_Set_buffers_V(&accel_t, (u64)buffer_t);
	XAccel_t_Set_params_V(&accel_t, (u64)params_t);
	XAccel_t_Set_probs(&accel_t, (u64)probs_t);

	float threshold = 9999;

	XTime start, end;
	int time_ms;
	XTime_StartTimer();
	XTime_GetTime(&start);

	xil_printf("Running network for %d images\n", NUM_TESTS);

	int num_correct = 0;
	float probs[1000];
	float max, max2, diff;
	int argmax, argmax2, label;
	int idx_t = 0;
	int idx_16 = 1;
	int next_idx = 2;
	bool done_accel_16 = false;
	bool done_accel_t = false;
	bool done_16 = false;
	bool done_t = false;
	std::queue<int> image_queue;
	int num_pushed = 0;

	XAccel_16_Set_image_V(&accel_16, (u64)images[idx_16]);
	Xil_DCacheFlush();
	XAccel_16_Start(&accel_16);

	XAccel_t_Set_image_V(&accel_t, (u64)images[idx_t]);
	Xil_DCacheFlush();
	XAccel_t_Start(&accel_t);

	while(!done_16 || !done_t) {
		while(!done_accel_16 && !done_accel_t) {
			done_accel_16 = XAccel_16_IsDone(&accel_16);
			done_accel_t = XAccel_t_IsDone(&accel_t);
		}
		if(done_accel_16) {
			done_accel_16 = false;
			for(int co = 0; co < 1000; co++) {
				probs[co] = probs_16[co];
			}

			max = probs[0];
			argmax = 0;
			for(int i = 1; i < 1000; i++) {
				if(max < probs[i]) {
					max = probs[i];
					argmax = i;
				}
			}

			probs[argmax] = -999;
			max2 = probs[0];
			argmax2 = 0;
			for(int i = 1; i < 1000; i++) {
				if(max2 < probs[i]) {
					max2 = probs[i];
					argmax2 = i;
				}
			}

			diff = max - max2;
			label = labels[idx_16];
			//xil_printf("[16] Image %d: Label: %d, Output: %d %d, Diff: %d\n", idx_16, label, argmax, argmax2, (int) (diff*1000));

			if(argmax == label) {
				num_correct++;
			}

			if(image_queue.empty()) {
				if(next_idx >= NUM_TESTS) {
					done_16 = true;
				}
				else {
					idx_16 = next_idx;
					next_idx++;
				}
			}
			else {
				idx_16 = image_queue.front();
				image_queue.pop();
			}

			if(!done_16) {
				XAccel_16_Set_image_V(&accel_16, (u64)images[idx_16]);
				Xil_DCacheFlush();
				XAccel_16_Start(&accel_16);
			}
		}
		else if(done_accel_t) {
			done_accel_t = false;
			for(int co = 0; co < 1000; co++) {
				probs[co] = probs_t[co];
			}

			max = probs[0];
			argmax = 0;
			for(int i = 1; i < 1000; i++) {
				if(max < probs[i]) {
					max = probs[i];
					argmax = i;
				}
			}

			probs[argmax] = -999;
			max2 = probs[0];
			argmax2 = 0;
			for(int i = 1; i < 1000; i++) {
				if(max2 < probs[i]) {
					max2 = probs[i];
					argmax2 = i;
				}
			}

			diff = max - max2;
			label = labels[idx_t];
			//xil_printf("[T] Image %d: Label: %d, Output: %d %d, Diff: %d\n", idx_t, label, argmax, argmax2, (int) (diff*1000));

			if(diff < threshold) {
				image_queue.push(idx_t);
				//xil_printf("  Pushed\n");
				num_pushed++;
			}
			else {
				if(argmax == label) {
					num_correct++;
				}
			}

			if(next_idx >= NUM_TESTS) {
				done_t = true;
			}
			else {
				idx_t = next_idx;
				next_idx++;
			}

			if(!done_t) {
				XAccel_t_Set_image_V(&accel_t, (u64)images[idx_t]);
				Xil_DCacheFlush();
				XAccel_t_Start(&accel_t);
			}
		}
	}

	XTime_GetTime(&end);
	time_ms = (end - start) / (COUNTS_PER_SECOND / 1000);
	xil_printf("Threshold: %d.%d\n%d images pushed\n", (int) threshold, ((int) (threshold * 10)) % 10, num_pushed);
	xil_printf("Accuracy: %d out of %d images\n", num_correct, NUM_TESTS);
	xil_printf("Time: %d.%3d s\n\n", time_ms / 1000, time_ms % 1000);

	return 0;
}

#endif

#if TEST_METHOD == TEST_PARALLEL_8_T

#include "xaccel_8.h"
#include "xaccel_t.h"

int main()
{
	xil_printf("Reading files\n");
	if(read_files()) {
		xil_printf("Read files failed\n");
		return -1;
	}


	xil_printf("Getting accelerators\n");
	int status;
	XAccel_8 accel_16;
	XAccel_8_Config* config_16;
	config_16 = XAccel_8_LookupConfig(XPAR_ACCEL_8_0_DEVICE_ID);
	if(!config_16) return -1;
	status = XAccel_8_CfgInitialize(&accel_16, config_16);
	if(status != XST_SUCCESS) return -1;

	XAccel_t accel_t;
	XAccel_t_Config* config_t;
	config_t = XAccel_t_LookupConfig(XPAR_ACCEL_T_0_DEVICE_ID);
	if(!config_t) return -1;
	status = XAccel_t_CfgInitialize(&accel_t, config_t);
	if(status != XST_SUCCESS) return -1;

	XAccel_8_Set_buffers_V(&accel_16, (u64)buffer_16);
	XAccel_8_Set_params_V(&accel_16, (u64)params_8);
	XAccel_8_Set_probs(&accel_16, (u64)probs_16);

	XAccel_t_Set_buffers_V(&accel_t, (u64)buffer_t);
	XAccel_t_Set_params_V(&accel_t, (u64)params_t);
	XAccel_t_Set_probs(&accel_t, (u64)probs_t);

	float threshold = 0;

	for(threshold = 0; threshold < 3.1; threshold += 0.1) {

	XTime start, end;
	int time_ms;
	XTime_StartTimer();
	XTime_GetTime(&start);

	xil_printf("Running network for %d images\n", NUM_TESTS);

	int num_correct = 0;
	float probs[1000];
	float max, max2, diff;
	int argmax, argmax2, label;
	int idx_t = 0;
	int idx_16 = 1;
	int next_idx = 2;
	bool done_accel_16 = false;
	bool done_accel_t = false;
	bool done_16 = false;
	bool done_t = false;
	std::queue<int> image_queue;
	int num_pushed = 0;

	XAccel_8_Set_image_V(&accel_16, (u64)images[idx_16]);
	Xil_DCacheFlush();
	XAccel_8_Start(&accel_16);

	XAccel_t_Set_image_V(&accel_t, (u64)images[idx_t]);
	Xil_DCacheFlush();
	XAccel_t_Start(&accel_t);

	while(!done_16 || !done_t) {
		while(!done_accel_16 && !done_accel_t) {
			done_accel_16 = XAccel_8_IsDone(&accel_16);
			done_accel_t = XAccel_t_IsDone(&accel_t);
		}
		if(done_accel_16) {
			done_accel_16 = false;
			for(int co = 0; co < 1000; co++) {
				probs[co] = probs_16[co];
			}

			max = probs[0];
			argmax = 0;
			for(int i = 1; i < 1000; i++) {
				if(max < probs[i]) {
					max = probs[i];
					argmax = i;
				}
			}

			probs[argmax] = -999;
			max2 = probs[0];
			argmax2 = 0;
			for(int i = 1; i < 1000; i++) {
				if(max2 < probs[i]) {
					max2 = probs[i];
					argmax2 = i;
				}
			}

			diff = max - max2;
			label = labels[idx_16];
			//xil_printf("[8] Image %d: Label: %d, Output: %d %d, Diff: %d\n", idx_16, label, argmax, argmax2, (int) (diff*1000));

			if(argmax == label) {
				num_correct++;
			}

			if(image_queue.empty()) {
				if(next_idx >= NUM_TESTS) {
					done_16 = true;
				}
				else {
					idx_16 = next_idx;
					next_idx++;
				}
			}
			else {
				idx_16 = image_queue.front();
				image_queue.pop();
			}

			if(!done_16) {
				XAccel_8_Set_image_V(&accel_16, (u64)images[idx_16]);
				Xil_DCacheFlush();
				XAccel_8_Start(&accel_16);
			}
		}
		else if(done_accel_t) {
			done_accel_t = false;
			for(int co = 0; co < 1000; co++) {
				probs[co] = probs_t[co];
			}

			max = probs[0];
			argmax = 0;
			for(int i = 1; i < 1000; i++) {
				if(max < probs[i]) {
					max = probs[i];
					argmax = i;
				}
			}

			probs[argmax] = -999;
			max2 = probs[0];
			argmax2 = 0;
			for(int i = 1; i < 1000; i++) {
				if(max2 < probs[i]) {
					max2 = probs[i];
					argmax2 = i;
				}
			}

			diff = max - max2;
			label = labels[idx_t];
			//xil_printf("[T] Image %d: Label: %d, Output: %d %d, Diff: %d\n", idx_t, label, argmax, argmax2, (int) (diff*1000));

			if(diff < threshold) {
				image_queue.push(idx_t);
				//xil_printf("  Pushed\n");
				num_pushed++;
			}
			else {
				if(argmax == label) {
					num_correct++;
				}
			}

			if(next_idx >= NUM_TESTS) {
				done_t = true;
			}
			else {
				idx_t = next_idx;
				next_idx++;
			}

			if(!done_t) {
				XAccel_t_Set_image_V(&accel_t, (u64)images[idx_t]);
				Xil_DCacheFlush();
				XAccel_t_Start(&accel_t);
			}
		}
	}

	XTime_GetTime(&end);
	time_ms = (end - start) / (COUNTS_PER_SECOND / 1000);
	xil_printf("Threshold: %d.%d\n%d images pushed\n", (int) threshold, ((int) (threshold * 10)) % 10, num_pushed);
	xil_printf("Accuracy: %d out of %d images\n", num_correct, NUM_TESTS);
	xil_printf("Time: %d.%3d s\n\n", time_ms / 1000, time_ms % 1000);

	}

	return 0;
}

#endif

#if TEST_METHOD == TEST_16BIT

#include "xaccel_16.h"

int main()
{
	init_platform();

	xil_printf("Reading files\n");
	if(read_files()) {
		xil_printf("Read files failed\n");
		return -1;
	}

	xil_printf("Getting accelerators\n");
	int status;
	XAccel_16 accel_16;
	XAccel_16_Config* config_16;
	config_16 = XAccel_16_LookupConfig(XPAR_ACCEL_16_0_DEVICE_ID);
	if(!config_16) return -1;
	status = XAccel_16_CfgInitialize(&accel_16, config_16);
	if(status != XST_SUCCESS) return -1;

	for(int l = 0; l < NUM_LAYERS; l++) {
		layer_info(info_16[l], l, false);
	}

	XAccel_16_Set_buffers_V(&accel_16, (u64)buffer_16);
	XAccel_16_Set_params_V(&accel_16, (u64)params_16);
	XAccel_16_Set_probs(&accel_16, (u64)probs_16);
	XAccel_16_Set_info_r(&accel_16, (u64)info_16);

	XTime start, end;
	int time_ms;
	XTime_StartTimer();
	XTime_GetTime(&start);

	XAccel_16_IsReady(&accel_16);
	xil_printf("Running network for %d images\n", NUM_TESTS);

	int num_correct = 0;
	float probs[1000];
	float max, max2, diff;
	int argmax, argmax2, label;
	for(int idx = 0; idx < NUM_TESTS; idx++) {
		XAccel_16_Set_image_V(&accel_16, (u64)images[idx]);
		info_16[1][0] = -1;
		for(int l = 0; l < NUM_LAYERS; l++) {
			layer_info(info_16[0], l, false);
			Xil_DCacheFlush();
			XAccel_16_Start(&accel_16);
			while(!XAccel_16_IsDone(&accel_16));
		}
		/*Xil_DCacheFlush();
		XAccel_16_Start(&accel_16);
		while(!XAccel_16_IsDone(&accel_16));*/

		for(int co = 0; co < 1000; co++) {
			probs[co] = probs_16[co];
		}

		max = probs[0];
		argmax = 0;
		for(int i = 1; i < 1000; i++) {
			if(max < probs[i]) {
				max = probs[i];
				argmax = i;
			}
		}

		probs[argmax] = -999;
		max2 = probs[0];
		argmax2 = 0;
		for(int i = 1; i < 1000; i++) {
			if(max2 < probs[i]) {
				max2 = probs[i];
				argmax2 = i;
			}
		}

		diff = max - max2;
		label = labels[idx];
		if(argmax == label) {
			num_correct++;
		}
		xil_printf("Image %d: Label: %d, Output: %d %d, Diff: %d, Acc: %d\n", idx, label, argmax, argmax2, (int) (diff*1000), num_correct);
	}

	XTime_GetTime(&end);
	time_ms = (end - start) / (COUNTS_PER_SECOND / 1000);
	xil_printf("Accuracy: %d out of %d images\n", num_correct, NUM_TESTS);
	xil_printf("Time: %d.%3d s\n", (time_ms / 1000), (time_ms % 1000));

	cleanup_platform();
	return 0;
}

#endif

#if TEST_METHOD == TEST_8BIT

#include "xaccel_8.h"

int main()
{
	xil_printf("Reading files\n");
	if(read_files()) {
		xil_printf("Read files failed\n");
		return -1;
	}

	xil_printf("Getting accelerators\n");
	int status;
	XAccel_8 accel_8;
	XAccel_8_Config* config_8;
	config_8 = XAccel_8_LookupConfig(XPAR_ACCEL_8_0_DEVICE_ID);
	if(!config_8) return -1;
	status = XAccel_8_CfgInitialize(&accel_8, config_8);
	if(status != XST_SUCCESS) return -1;

	XAccel_8_Set_buffers_V(&accel_8, (u64)buffer_16);
	XAccel_8_Set_params_V(&accel_8, (u64)params_8);
	XAccel_8_Set_probs(&accel_8, (u64)probs_16);

	XTime start, end;
	int time_ms;
	XTime_StartTimer();
	XTime_GetTime(&start);

	xil_printf("Running network for %d images\n", NUM_TESTS);

	int num_correct = 0;
	float probs[1000];
	float max, max2, diff;
	int argmax, argmax2, label;
	for(int idx = 0; idx < NUM_TESTS; idx++) {
		XAccel_8_Set_image_V(&accel_8, (u64)images[idx]);
		Xil_DCacheFlush();
		XAccel_8_Start(&accel_8);
		while(!XAccel_8_IsDone(&accel_8));

		for(int co = 0; co < 1000; co++) {
			probs[co] = probs_16[co];
		}

		max = probs[0];
		argmax = 0;
		for(int i = 1; i < 1000; i++) {
			if(max < probs[i]) {
				max = probs[i];
				argmax = i;
			}
		}

		probs[argmax] = -999;
		max2 = probs[0];
		argmax2 = 0;
		for(int i = 1; i < 1000; i++) {
			if(max2 < probs[i]) {
				max2 = probs[i];
				argmax2 = i;
			}
		}

		diff = max - max2;
		label = labels[idx];
		if(argmax == label) {
			num_correct++;
		}
		xil_printf("Image %d: Label: %d, Output: %d %d, Diff: %d, Acc: %d\n", idx, label, argmax, argmax2, (int) (diff*1000), num_correct);
	}


	XTime_GetTime(&end);
	time_ms = (end - start) / (COUNTS_PER_SECOND / 1000);
	xil_printf("Accuracy: %d out of %d images\n", num_correct, NUM_TESTS);
	xil_printf("Time: %d.%3d s\n", (time_ms / 1000), (time_ms % 1000));

	return 0;
}

#endif

#if TEST_METHOD == TEST_TERNARY

#include "xaccel_t.h"

int main()
{
	xil_printf("Reading files\n");
	if(read_files()) {
		xil_printf("Read files failed\n");
		return -1;
	}

	xil_printf("Getting accelerators\n");
	int status;
	XAccel_t accel_t;
	XAccel_t_Config* config_t;
	config_t = XAccel_t_LookupConfig(XPAR_ACCEL_T_0_DEVICE_ID);
	if(!config_t) return -1;
	status = XAccel_t_CfgInitialize(&accel_t, config_t);
	if(status != XST_SUCCESS) return -1;

	for(int l = 0; l < NUM_LAYERS; l++) {
		layer_info(info_t[l], l, true);
	}

	XAccel_t_Set_buffers_V(&accel_t, (u64)buffer_t);
	XAccel_t_Set_params_V(&accel_t, (u64)params_t);
	XAccel_t_Set_probs(&accel_t, (u64)probs_t);
	XAccel_t_Set_info_r(&accel_t, (u64)info_t);

	XTime start, end;
	int time_ms;
	XTime_StartTimer();
	XTime_GetTime(&start);

	XAccel_t_IsReady(&accel_t);
	xil_printf("Running network for %d images\n", NUM_TESTS);

	int num_correct = 0;
	float probs[1000];
	float max, max2, diff;
	int argmax, argmax2, label;
	for(int idx = 0; idx < NUM_TESTS; idx++) {
		XAccel_t_Set_image_V(&accel_t, (u64)images[idx]);
		info_t[1][0] = -1;
		for(int l = 0; l < NUM_LAYERS; l++) {
			layer_info(info_t[0], l, true);
			Xil_DCacheFlush();
			XAccel_t_Start(&accel_t);
			while(!XAccel_t_IsDone(&accel_t));
		}

		for(int co = 0; co < 1000; co++) {
			probs[co] = probs_t[co];
		}

		max = probs[0];
		argmax = 0;
		for(int i = 1; i < 1000; i++) {
			if(max < probs[i]) {
				max = probs[i];
				argmax = i;
			}
		}

		probs[argmax] = -999;
		max2 = probs[0];
		argmax2 = 0;
		for(int i = 1; i < 1000; i++) {
			if(max2 < probs[i]) {
				max2 = probs[i];
				argmax2 = i;
			}
		}

		diff = max - max2;
		label = labels[idx];
		if(argmax == label) {
			num_correct++;
		}
		xil_printf("Image %d: Label: %d, Output: %d %d, Diff: %d, Acc: %d\n", idx, label, argmax, argmax2, (int) (diff*1000), num_correct);
	}

	XTime_GetTime(&end);
	time_ms = (end - start) / (COUNTS_PER_SECOND / 1000);
	xil_printf("Accuracy: %d out of %d images\n", num_correct, NUM_TESTS);
	xil_printf("Time: %d.%3d s\n", (time_ms / 1000), (time_ms % 1000));

	return 0;
}

#endif

