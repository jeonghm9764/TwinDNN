#include "net.h"

#define ABS(N) ((N<0)?(-N):(N))
#define BIN(N) ((N<0)?0:1)
#define TER(N) ((N==0)?0:(N>0?1:-1))

#define NUM_TESTS 1

float images_float[NUM_TESTS][3][224][224];
UFIX_64 images_fix[NUM_TESTS][226][226];

float* weights_float[NUM_LAYERS];
float* weights_scale[NUM_LAYERS];
float* bias_float[NUM_LAYERS];
FIX_256 params[NUM_PARAMS];

FIX_256 buffer[3][80][114][114];

float probs[1000];
float labels[NUM_TESTS];

int info[NUM_LAYERS*SIZE_INFO];

int read_images(char const* filename) {
    int ret;
    FILE *fil;
	fil = fopen(filename, "rb");
	if(!fil) return -1;
	for(int i = 0; i < NUM_TESTS; i++) {
		ret = fread((void*)images_float[i], sizeof(float), 3*224*224, fil);
		if(!ret) return -1;
	}
	ret = fclose(fil);
	if(ret) return -1;

	return 0;
}

int read_labels(char const* filename) {
	int ret;
	FILE *fil = 0;
	fil = fopen(filename, "rb");
	if(!fil) return -1;
	ret = fread((void*)labels, sizeof(float), NUM_TESTS, fil);
	if(!ret) return -1;
	ret = fclose(fil);
	if(ret) return -1;

	return 0;
}

int read_params(char const* filename) {
	int ret, size;
	FILE *fil = 0;
	fil = fopen(filename, "rb");
	if(!fil) return -1;
	for(int l = 0; l < NUM_LAYERS; l++) {
		size = CIN[l] * COUT[l] * KERNEL[l] * KERNEL[l];
		weights_float[l] = new float[size];
		fread((void*)(weights_float[l]), sizeof(float), size, fil);
		if(!ret) return -1;
		bias_float[l] = new float[COUT[l]];
		fread((void*)(bias_float[l]), sizeof(float), COUT[l], fil);
		if(!ret) return -1;
	}
	if(!ret) return -1;
	ret = fclose(fil);
	if(ret) return -1;

	return 0;
}

int reorder_images() {
	FIX_A pixel;

	for(int idx = 0; idx < NUM_TESTS; idx++) {
		for(int h = 0; h < 226; h++) {
			for(int w = 0; w < 226; w++) {
				images_fix[idx][h][w] = 0;
			}
		}
		for(int h = 0; h < 224; h++) {
			for(int w = 0; w < 224; w++) {
				for(int c = 0; c < 3; c++) {
					pixel = images_float[idx][c][h][w];
					images_fix[idx][h+1][w+1].range(BIT_A*c+BIT_A-1, BIT_A*c) = pixel.range(BIT_A-1, 0);
				}
			}
		}
	}

	return 0;
}

void get_weight_scales() {
	float w;
	int widx;
	for(int l = 1; l < NUM_LAYERS-1; l++) {
		weights_scale[l] = new float[COUT[l]];
		for(int co = 0; co < COUT[l]; co++) {
			widx = co*(CIN[l]*KERNEL[l]*KERNEL[l]);
			for(int offset = 0; offset < (CIN[l]*KERNEL[l]*KERNEL[l]); offset++) {
				if(weights_float[l][widx+offset] != 0) {
					weights_scale[l][co] = ABS(weights_float[l][widx+offset]);
					break;
				}
				if(offset == (CIN[l]*KERNEL[l]*KERNEL[l] - 1)) {
					weights_scale[l][co] = 0;
				}
			}

			//printf("%f\n", weights_scale[l][co]);
		}
	}
}

int reorder_params() {
	int idx = 0;
	int widx;
	FIX_W temp;
	FIX_2 w_t;

	get_weight_scales();

	// conv1 layer
	for(int co = 0; co < COUT[0]; co += 16) {
		params[idx] = 0;
		for(int coo = 0; coo < 16; coo++) {
			temp = bias_float[0][co+coo];
			params[idx].range(BIT_W*coo+BIT_W-1, BIT_W*coo) = temp.range(BIT_W-1, 0);
		}
		idx++;
		for(int ci = 0; ci < CIN[0]; ci += 16) {
			for(int h_in_mul = 0; h_in_mul < 224; h_in_mul += 112) {
				for(int w_in_mul = 0; w_in_mul < 224; w_in_mul += 112) {
					for(int w = 0; w < KERNEL[0]; w++) {
						for(int h = 0; h < KERNEL[0]; h++) {
							for(int coo = 0; coo < 16; coo++, idx++) {
								params[idx] = 0;
								for(int cii = 0; cii < CIN[0]; cii++) {
									widx = (co+coo)*(CIN[0]*KERNEL[0]*KERNEL[0])+(ci+cii)*(KERNEL[0]*KERNEL[0])+w*KERNEL[0]+h;
									temp = weights_float[0][widx];
									params[idx].range(BIT_W*cii+BIT_W-1, BIT_W*cii) = temp.range(BIT_W-1, 0);
								}
							}
						}
					}
				}
			}
		}
	}
	printf("{0, %d, ", idx);

	// other layers
	for(int l = 1; l < NUM_LAYERS-1; l++) {
		for(int co = 0; co < COUT[l]; co += 16) {
			params[idx] = 0;
			for(int coo = 0; coo < 16; coo++) {
				temp = bias_float[l][co+coo];
				params[idx].range(BIT_W*coo+BIT_W-1, BIT_W*coo) = temp.range(BIT_W-1, 0);
			}
			idx++;
			params[idx] = 0;
			for(int coo = 0; coo < 16; coo++) {
				temp = weights_scale[l][co+coo];
				params[idx].range(BIT_W*coo+BIT_W-1, BIT_W*coo) = temp.range(BIT_W-1, 0);
			}
			idx++;
			for(int ci = 0; ci < CIN[l]; ci += 16) {
				for(int w = 0; w < KERNEL[l]; w++) {
					for(int h = 0; h < KERNEL[l]; h++) {
						for(int coo = 0; coo < 16; coo++, idx++) {
							params[idx] = 0;
							if(CIN[l] != 1) {
								for(int cii = 0; cii < 16; cii++) {
									if(ci+cii < CIN[l]) {
										widx = (co+coo)*(CIN[l]*KERNEL[l]*KERNEL[l])+(ci+cii)*(KERNEL[l]*KERNEL[l])+w*KERNEL[l]+h;
										w_t = TER(weights_float[l][widx]);
									}
									else {
										w_t = 0;
									}
									params[idx].range(cii*2+1, cii*2) = w_t.range(1, 0);
								}
							}
							else { // if depthwise
								widx = (co+coo)*(KERNEL[l]*KERNEL[l])+w*KERNEL[l]+h;
								w_t = TER(weights_float[l][widx]);
								params[idx].range(coo*2+1, coo*2) = w_t.range(1, 0);
							}
						}
					}
				}
			}
		}
		printf("%d, ", idx);
	}

	// fc layer
	for(int co = 0; co < COUT[NUM_LAYERS-1]; co += 16, idx++) {
		params[idx] = 0;
		for(int coo = 0; coo < 16; coo++) {
			if(co+coo >= COUT[NUM_LAYERS-1]) break;
			temp = bias_float[NUM_LAYERS-1][co+coo];
			params[idx].range(BIT_W*coo+BIT_W-1, BIT_W*coo) = temp.range(BIT_W-1, 0);
		}
	}
	for(int ci = 0; ci < CIN[NUM_LAYERS-1]; ci += 16) {
		for(int co = 0; co < COUT[NUM_LAYERS-1]; co += 16) {
			for(int coo = 0; coo < 16; coo++, idx++) {
				params[idx] = 0;
				if(co+coo >= COUT[NUM_LAYERS-1]) continue;
				for(int cii = 0; cii < 16; cii++) {
					widx = (co+coo)*CIN[NUM_LAYERS-1]+(ci+cii);
					temp = weights_float[NUM_LAYERS-1][widx];
					params[idx].range(BIT_W*cii+BIT_W-1, BIT_W*cii) = temp.range(BIT_W-1, 0);
				}
			}
		}
	}
	printf("%d}\n", idx);

	return 0;
}

int write_reordered_images(char const* filename) {
    int ret;
    FILE *fil = 0;
	fil = fopen(filename, "wb");
	if(!fil) return -1;
	ret = fwrite((void*)images_fix, sizeof(UFIX_64), NUM_TESTS*226*226, fil);
	if(!ret) return -1;
	ret = fclose(fil);
	if(ret) return -1;

	return 0;
}

int write_reordered_params(char const* filename) {
    int ret;
    FILE *fil = 0;
	fil = fopen(filename, "wb");
	if(!fil) return -1;
	ret = fwrite((void*)params, sizeof(FIX_256), NUM_PARAMS, fil);
	if(!ret) return -1;
	ret = fclose(fil);
	if(ret) return -1;

	return 0;
}

int read_reordered_images(char const* filename) {
    int ret;
    FILE *fil = 0;
	fil = fopen(filename, "rb");
	if(!fil) return -1;
	ret = fread((void*)images_fix, sizeof(UFIX_64), NUM_TESTS*226*226, fil);
	if(!ret) return -1;
	ret = fclose(fil);
	if(ret) return -1;

	return 0;
}

int read_reordered_params(char const* filename) {
    int ret;
    FILE *fil = 0;
	fil = fopen(filename, "rb");
	if(!fil) return -1;
	ret = fread((void*)params, sizeof(FIX_256), NUM_PARAMS, fil);
	if(!ret) return -1;
	ret = fclose(fil);
	if(ret) return -1;

	return 0;
}

void layer_info(int info[12], int layer) {
	info[0] = layer;
	info[1] = INPUT[layer];
	info[2] = ACCUM[layer];
	info[3] = OUTPUT[layer];
	info[4] = CIN[layer];
	info[5] = COUT[layer];
	info[6] = DIM[layer];
	info[7] = KERNEL[layer];
	info[8] = STRIDE[layer];
	info[9] = IDX[layer];
	info[10] = RELU[layer];
	info[11] = ADD[layer];
}

int main() {
	printf("Reading labels\n");
	if(read_labels("labels_1000.dat")) return -1;
	printf("Reading images\n");
	if(read_images("images_1000.dat")) return -1;
	printf("Reading parameters\n");
	if(read_params("mobilenet_t.dat")) return -1;
	printf("Reordering images\n");
	if(reorder_images()) return -1;
	printf("Reordering parameters\n");
	if(reorder_params()) return -1;
	printf("Writing reordered images\n");
	if(write_reordered_images("images_reordered.dat")) return -1;
	printf("Writing reordered parameters\n");
	if(write_reordered_params("mobilenet_t_reordered.dat")) return -1;
	/*printf("Reading reordered images\n");
	if(read_reordered_images("images_reordered.dat")) return -1;
	printf("Reading reordered parameters\n");
	if(read_reordered_params("mobilenet_t_reordered.dat")) return -1;*/

	printf("Running network for %d images\n", NUM_TESTS);

	for(int l = 0; l < NUM_LAYERS; l++) {
		layer_info(info+l*SIZE_INFO, l);
	}

	float max, max2, diff;
	int argmax, argmax2, label;
	int num_correct = 0;
	for(int idx = 0; idx < NUM_TESTS; idx++) {
		info[SIZE_INFO] = -1;
		for(int l = 0; l < NUM_LAYERS; l++) {
			layer_info(info, l);
			accel_t(images_fix[idx], buffer, params, probs, info);
		}
		//accel_t(images_fix[idx], buffer, params, probs, info);

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
		if(argmax == labels[idx]) {
			num_correct++;
		}
		printf("Image %d: Label: %d, Output: %d %d, Diff: %d, Acc: %d\n", idx, label, argmax, argmax2, (int) (diff*1000), num_correct);
	}

	return 0;
}

