#define HALF_ENABLE_CPP11_CMATH 0

#define NUM_PARAMS_16 780399
#define NUM_PARAMS_8 415471
#define NUM_PARAMS_T 780703

#define NUM_TESTS 1000

#define TEST_TERNARY 0
#define TEST_16BIT 1
#define TEST_8BIT 2
#define TEST_PARALLEL_16_T 3
#define TEST_PARALLEL_8_T 4
#define TEST_METHOD TEST_8BIT

#include <queue>
#include "xil_printf.h"
#include "xparameters.h"
#include "ff.h"
#include "xil_cache.h"
#include "xtime_l.h"
#include "xstatus.h"

uint64_t buffer_16[4*3*32*64*64];
uint64_t buffer_t[4*3*32*64*64];

uint64_t params_16[4*NUM_PARAMS_16];
uint64_t params_8[4*NUM_PARAMS_8];
uint64_t params_t[4*NUM_PARAMS_T];

uint32_t images[NUM_TESTS][230][230];

float probs_16[1000];
float probs_t[1000];
float labels[NUM_TESTS];

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
	Res = f_read(&fil, (void*)images, sizeof(uint32_t)*NUM_TESTS*230*230, &NumBytesRead);
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
	Res = f_read(&fil, (void*)params_16, sizeof(uint64_t)*4*NUM_PARAMS_16, &NumBytesRead);
	if(Res) return XST_FAILURE;
	Res = f_close(&fil);

	Res = f_open(&fil, "params_8.dat", FA_READ);
	if(Res) return XST_FAILURE;
	Res = f_lseek(&fil, 0);
	if(Res) return XST_FAILURE;
	Res = f_read(&fil, (void*)params_8, sizeof(uint64_t)*4*NUM_PARAMS_8, &NumBytesRead);
	if(Res) return XST_FAILURE;
	Res = f_close(&fil);

	Res = f_open(&fil, "params_t.dat", FA_READ);
	if(Res) return XST_FAILURE;
	Res = f_lseek(&fil, 0);
	if(Res) return XST_FAILURE;
	Res = f_read(&fil, (void*)params_t, sizeof(uint64_t)*4*NUM_PARAMS_T, &NumBytesRead);
	if(Res) return XST_FAILURE;
	Res = f_close(&fil);

	return 0;
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
			xil_printf("[8] Image %d: Label: %d, Output: %d %d, Diff: %d\n", idx_16, label, argmax, argmax2, (int) (diff*1000));

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
			xil_printf("[T] Image %d: Label: %d, Output: %d %d, Diff: %d\n", idx_t, label, argmax, argmax2, (int) (diff*1000));

			if(diff < THRESHOLD) {
				image_queue.push(idx_t);
				xil_printf("Pushed Image %d\n", idx_t);
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
	xil_printf("Accuracy: %d out of %d images\n", num_correct, NUM_TESTS);
	xil_printf("Time: %d.%3d s", (time_ms / 1000), (time_ms % 1000));

	return 0;
}

#endif

#if TEST_METHOD == TEST_16BIT

#include "xaccel_16.h"

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

	XAccel_16_Set_buffers_V(&accel_16, (u64)buffer_16);
	XAccel_16_Set_params_V(&accel_16, (u64)params_16);
	XAccel_16_Set_probs(&accel_16, (u64)probs_16);

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
		XAccel_16_Set_image_V(&accel_16, (u64)images[idx]);
		Xil_DCacheFlush();
		XAccel_16_Start(&accel_16);
		while(!XAccel_16_IsDone(&accel_16));

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

	XAccel_t_Set_buffers_V(&accel_t, (u64)buffer_t);
	XAccel_t_Set_params_V(&accel_t, (u64)params_t);
	XAccel_t_Set_probs(&accel_t, (u64)probs_t);

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
		XAccel_t_Set_image_V(&accel_t, (u64)images[idx]);
		Xil_DCacheFlush();
		XAccel_t_Start(&accel_t);
		while(!XAccel_t_IsDone(&accel_t));

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

