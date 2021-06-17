#include "net.h"

void accel_t(
		UFIX_64 image[226][226],
		FIX_256 buffers[3][80][114][114],
		FIX_256 params[NUM_PARAMS],
		float probs[1000],
		int info[NUM_LAYERS*SIZE_INFO]
		) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=IMAGE
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=PARAM
#pragma HLS INTERFACE m_axi port=buffers offset=slave bundle=BUFFER
#pragma HLS INTERFACE m_axi port=probs offset=slave bundle=PROB
#pragma HLS INTERFACE m_axi port=info offset=slave bundle=INFO
#pragma HLS INTERFACE s_axilite port=return
	int info_buf[NUM_LAYERS*SIZE_INFO];
	for(int l = 0; l < NUM_LAYERS*SIZE_INFO; l++) {
#pragma HLS PIPELINE
		info_buf[l] = info[l];
	}
	for(int i = 0; i < NUM_LAYERS*SIZE_INFO; i += SIZE_INFO) {
		if(info_buf[i] == -1) break;
		layer(image, buffers[info_buf[i+1]], params, buffers[info_buf[i+2]], buffers[info_buf[i+3]], probs, info_buf+i);
	}
}

void layer(
		UFIX_64 image[226][226],
		FIX_256 input[80][114][114],
		FIX_256 params[NUM_PARAMS],
		FIX_256 accum[80][114][114],
		FIX_256 output[80][114][114],
		float probs[1000],
		int layer_info[SIZE_INFO]
		) {
	FIX_2 weights_t[16][16];
#pragma HLS ARRAY_PARTITION variable=weights_t complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights_t complete dim=2
	FIX_W weights[16][16];
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
	FIX_W weights_scale[16];
#pragma HLS ARRAY_PARTITION variable=weights_scale complete dim=1
	FIX_W bias[16];
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1
	FIX_A input_buf[16][128][128];
#pragma HLS ARRAY_PARTITION variable=input_buf complete dim=1
	FIX_ACC output_buf[16][128][128];
#pragma HLS ARRAY_PARTITION variable=output_buf complete dim=1

	FIX_256 temp_256;
	UFIX_64 temp_64;
	FIX_A temp_act[16];
#pragma HLS ARRAY_PARTITION variable=temp_act complete dim=1
	FIX_ACC temp_acc[16];
#pragma HLS ARRAY_PARTITION variable=temp_acc complete dim=1

	int layer = layer_info[0];
	int cin = layer_info[4];
	int cout = layer_info[5];
	int dim = layer_info[6];
	int kernel = layer_info[7];
	int stride = layer_info[8];
	int param_idx = layer_info[9];
	int relu = layer_info[10];
	int add = layer_info[11];
	int outdim = dim * stride + 2;
	int h_idx, w_idx, ci_mod;

	// conv1 layer
	if(layer == 0) {
		for(int co = 0; co < cout; co += 16) {
			temp_256 = params[param_idx++];
			for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}

			for(int h_in_mul = 0, h_out_mul = 0; h_in_mul < 224; h_in_mul += 112, h_out_mul += 56) {
				for(int w_in_mul = 0, w_out_mul = 0; w_in_mul < 224; w_in_mul += 112, w_out_mul += 56) {
					// load image
					for(int h = 0; h < 114; h++) {
						for(int w = 0; w < 114; w++) {
#pragma HLS PIPELINE
							temp_64 = image[h_in_mul+h][w_in_mul+w];
							for(int cii = 0; cii < 3; cii++) {
#pragma HLS UNROLL
								input_buf[cii][h][w].range(BIT_A-1, 0) = temp_64.range(BIT_A*cii+BIT_A-1, BIT_A*cii);
							}
						}
					}

					for(int h = 0; h < 56; h++) {
						for(int w = 0; w < 56; w++) {
#pragma HLS PIPELINE
							for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								output_buf[coo][h][w] = bias[coo];
							}
						}
					}

					for(int hw = 0; hw < 57; hw++) {
#pragma HLS PIPELINE
						for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
							output_buf[coo][hw][56] = 0;
							output_buf[coo][56][hw] = 0;
						}
					}


					for(int i = 0; i < 3; i++) {
						for(int j = 0; j < 3; j++) {
							for(int wi = 0; wi < 16; wi++) {
#pragma HLS PIPELINE
								temp_256 = params[param_idx++];
								for(int wj = 0; wj < 16; wj++) {
#pragma HLS UNROLL
									weights[wi][wj].range(BIT_W-1, 0) = temp_256.range(BIT_W*wj+BIT_W-1, BIT_W*wj);
								}
							}

							h_idx = i;
							for(int h = 0; h < 56; h++, h_idx += 2) {
								w_idx = j;
								for(int w = 0; w < 56; w++, w_idx += 2) {
									for(int coo = 0; coo < 16; coo++) {
#pragma HLS PIPELINE
										output_buf[coo][h][w] = output_buf[coo][h][w] +
												weights[coo][0] * input_buf[0][h_idx][w_idx] +
												weights[coo][1] * input_buf[1][h_idx][w_idx] +
												weights[coo][2] * input_buf[2][h_idx][w_idx];
									}
								}
							}
						}
					}

					for(int h = 0; h < 56; h++) {
						for(int w = 0; w < 56; w++) {
#pragma HLS PIPELINE
							for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								if(output_buf[coo][h][w] < 0)
									output_buf[coo][h][w] = 0;
							}
						}
					}

					for(int h = 0; h < 56; h++) {
						for(int w = 0; w < 56; w++) {
#pragma HLS PIPELINE
							for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								temp_act[coo] = output_buf[coo][h][w];
								temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo) = temp_act[coo].range(BIT_A-1, 0);
							}
							output[co>>4][h+h_out_mul+1][w+w_out_mul+1] = temp_256;
						}
					}
				}
			}
		}

		for(int co = 0; co < cout; co += 16) {
			for(int hw = 0; hw < 114; hw++) {
				output[co>>4][0][hw] = 0;
				output[co>>4][113][hw] = 0;
				output[co>>4][hw][0] = 0;
				output[co>>4][hw][113] = 0;
			}
		}
	}

	// other layers
	else if(layer < NUM_LAYERS-1) {
		for(int co = 0; co < cout; co += 16) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
			temp_256 = params[param_idx++];
			for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}
			temp_256 = params[param_idx++];
			for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				weights_scale[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}

			for(int h = 0; h < dim + 2; h++) {
#pragma HLS LOOP_TRIPCOUNT min=114 max=114
				for(int w = 0; w < dim + 2; w++) {
#pragma HLS LOOP_TRIPCOUNT min=114 max=114
#pragma HLS PIPELINE
					for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
						output_buf[coo][h][w] = 0;
					}
				}
			}

			if(co == 0) { // no function, do not remove
				output[layer][0][0] = 0;
			}

			for(int ci = 0; ci < cin; ci += 16) {
#pragma HLS LOOP_TRIPCOUNT min=2 max=2
				ci_mod = (cin == 1) ? co : ci;
				for(int h = 0; h < outdim; h++) {
#pragma HLS LOOP_TRIPCOUNT min=112 max=112
					for(int w = 0; w < outdim; w++) {
#pragma HLS LOOP_TRIPCOUNT min=112 max=112
#pragma HLS PIPELINE
						temp_256 = input[ci_mod>>4][h][w];
						for(int cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
							input_buf[cii][h][w].range(BIT_A-1, 0) = temp_256.range(BIT_A*cii+BIT_A-1, BIT_A*cii);
						}
					}
				}

				for(int i = 0; i < kernel; i++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
					for(int j = 0; j < kernel; j++) {
#pragma HLS LOOP_TRIPCOUNT min=3 max=3
						for(int wi = 0; wi < 16; wi++) {
#pragma HLS PIPELINE
							temp_256 = params[param_idx++];
							for(int wj = 0; wj < 16; wj++) {
#pragma HLS UNROLL
								weights_t[wi][wj].range(1, 0) = temp_256.range(wj*2+1, wj*2);
							}
						}

						h_idx = (kernel == 1) ? 1 : i;
						for(int h = 1; h < dim + 1; h++, h_idx += stride) {
#pragma HLS LOOP_TRIPCOUNT min=112 max=112
							w_idx = (kernel == 1) ? 1 : j;
							for(int w = 1; w < dim + 1; w++, w_idx += stride) {
#pragma HLS LOOP_TRIPCOUNT min=112 max=112
#pragma HLS PIPELINE
								for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
									output_buf[coo][h][w] = output_buf[coo][h][w] +
											compute_engine_16_t(weights_t[coo][0], input_buf[0][h_idx][w_idx],
											weights_t[coo][1], input_buf[1][h_idx][w_idx],
											weights_t[coo][2], input_buf[2][h_idx][w_idx],
											weights_t[coo][3], input_buf[3][h_idx][w_idx],
											weights_t[coo][4], input_buf[4][h_idx][w_idx],
											weights_t[coo][5], input_buf[5][h_idx][w_idx],
											weights_t[coo][6], input_buf[6][h_idx][w_idx],
											weights_t[coo][7], input_buf[7][h_idx][w_idx],
											weights_t[coo][8], input_buf[8][h_idx][w_idx],
											weights_t[coo][9], input_buf[9][h_idx][w_idx],
											weights_t[coo][10], input_buf[10][h_idx][w_idx],
											weights_t[coo][11], input_buf[11][h_idx][w_idx],
											weights_t[coo][12], input_buf[12][h_idx][w_idx],
											weights_t[coo][13], input_buf[13][h_idx][w_idx],
											weights_t[coo][14], input_buf[14][h_idx][w_idx],
											weights_t[coo][15], input_buf[15][h_idx][w_idx]);
								}
							}
						}
					}
				}
			}

			for(int h = 1; h < dim + 1; h++) {
#pragma HLS LOOP_TRIPCOUNT min=112 max=112
				for(int w = 1; w < dim + 1; w++) {
#pragma HLS LOOP_TRIPCOUNT min=112 max=112
#pragma HLS PIPELINE
					for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
						output_buf[coo][h][w] = output_buf[coo][h][w] * weights_scale[coo] + bias[coo];
					}
				}
			}

			if(add == 1) {
				for(int h = 1; h < dim + 1; h++) {
#pragma HLS LOOP_TRIPCOUNT min=112 max=112
					for(int w = 1; w < dim + 1; w++) {
#pragma HLS LOOP_TRIPCOUNT min=112 max=112
#pragma HLS PIPELINE
						temp_256 = accum[co>>4][h][w];
						for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
							temp_act[coo].range(BIT_A-1, 0) = temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo);
						}
						for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
							output_buf[coo][h][w] = output_buf[coo][h][w] + temp_act[coo];
						}
					}
				}
			}

			if(relu == 1) {
				for(int h = 1; h < dim + 1; h++) {
#pragma HLS LOOP_TRIPCOUNT min=112 max=112
					for(int w = 1; w < dim + 1; w++) {
#pragma HLS LOOP_TRIPCOUNT min=112 max=112
#pragma HLS PIPELINE
						for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
							if(output_buf[coo][h][w] < 0)
								output_buf[coo][h][w] = 0;
						}
					}
				}
			}

			for(int h = 0; h < dim + 2; h++) {
#pragma HLS LOOP_TRIPCOUNT min=114 max=114
				for(int w = 0; w < dim + 2; w++) {
#pragma HLS LOOP_TRIPCOUNT min=114 max=114
#pragma HLS PIPELINE
					for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
						temp_act[coo] = output_buf[coo][h][w];
					}
					for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
						temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo) = temp_act[coo].range(BIT_A-1, 0);
					}
					output[co>>4][h][w] = temp_256;
				}
			}
		}
	}

	// fc layer
	else {
		for(int co = 0; co < 1000; co += 16) {
#pragma HLS PIPELINE
			temp_256 = params[param_idx++];
			for(int coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
				output_buf[coo][co>>5][co%32] = bias[coo];
			}
		}

		for(int ci = 0; ci < cin; ci += 16) {
			for(int cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
				temp_acc[cii] = 0;
			}
			for(int h = 1; h < 8; h++) {
				for(int w = 1; w < 8; w++) {
#pragma HLS PIPELINE
					temp_256 = input[ci>>4][h][w];
					for(int cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
						temp_act[cii].range(BIT_A-1, 0) = temp_256.range(BIT_A*cii+BIT_A-1, BIT_A*cii);
						temp_acc[cii] = temp_acc[cii] + temp_act[cii];
					}
				}
			}
			for(int cii = 0; cii < 16; cii++) {
#pragma HLS PIPELINE
				temp_act[cii] = temp_acc[cii] / 49;
			}

			for(int co = 0; co < 1000; co += 16) {
				for(int wi = 0; wi < 16; wi++) {
#pragma HLS PIPELINE
					temp_256 = params[param_idx++];
					for(int wj = 0; wj < 16; wj++) {
#pragma HLS UNROLL
						weights[wi][wj].range(BIT_W-1, 0) = temp_256.range(BIT_W*wj+BIT_W-1, BIT_W*wj);
					}
				}
				for(int cii = 0; cii < 16; cii++) {
					for(int coo = 0; coo < 16; coo++) {
#pragma HLS PIPELINE
						output_buf[coo][co>>5][co%32] = output_buf[coo][co>>5][co%32] + weights[coo][cii] * temp_act[cii];
					}
				}
			}
		}

		for(int co = 0; co < 1000; co += 16) {
			for(int coo = 0; coo < 16; coo++) {
#pragma HLS PIPELINE
				if(co+coo < 1000) {
					probs[co+coo] = output_buf[coo][co>>5][co%32];
				}
			}
		}
	}

	//printf("%d\n", param_idx);
}

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
		FIX_2 w15, FIX_A a15) {
	FIX_32 mul0, mul1, mul2, mul3, mul4, mul5, mul6, mul7;
	FIX_32 mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	FIX_32 add0_0, add0_1, add0_2, add0_3, add0_4, add0_5, add0_6, add0_7;
	FIX_32 add1_0, add1_1, add1_2, add1_3;
	FIX_32 add2_0, add2_1;
	FIX_32 add3_0;

	switch(w0) {
	case -1: mul0 = -a0; break;
	case 0: mul0 = 0; break;
	case 1: mul0 = a0; break;
	case 2: mul0 = 0; break;
	default: mul0 = 0; break;
	}
	switch(w1) {
	case -1: mul1 = -a1; break;
	case 0: mul1 = 0; break;
	case 1: mul1 = a1; break;
	case 2: mul1 = 0; break;
	default: mul1 = 0; break;
	}
	switch(w2) {
	case -1: mul2 = -a2; break;
	case 0: mul2 = 0; break;
	case 1: mul2 = a2; break;
	case 2: mul2 = 0; break;
	default: mul2 = 0; break;
	}
	switch(w3) {
	case -1: mul3 = -a3; break;
	case 0: mul3 = 0; break;
	case 1: mul3 = a3; break;
	case 2: mul3 = 0; break;
	default: mul3 = 0; break;
	}
	switch(w4) {
	case -1: mul4 = -a4; break;
	case 0: mul4 = 0; break;
	case 1: mul4 = a4; break;
	case 2: mul4 = 0; break;
	default: mul4 = 0; break;
	}
	switch(w5) {
	case -1: mul5 = -a5; break;
	case 0: mul5 = 0; break;
	case 1: mul5 = a5; break;
	case 2: mul5 = 0; break;
	default: mul5 = 0; break;
	}
	switch(w6) {
	case -1: mul6 = -a6; break;
	case 0: mul6 = 0; break;
	case 1: mul6 = a6; break;
	case 2: mul6 = 0; break;
	default: mul6 = 0; break;
	}
	switch(w7) {
	case -1: mul7 = -a7; break;
	case 0: mul7 = 0; break;
	case 1: mul7 = a7; break;
	case 2: mul7 = 0; break;
	default: mul7 = 0; break;
	}
	switch(w8) {
	case -1: mul8 = -a8; break;
	case 0: mul8 = 0; break;
	case 1: mul8 = a8; break;
	case 2: mul8 = 0; break;
	default: mul8 = 0; break;
	}
	switch(w9) {
	case -1: mul9 = -a9; break;
	case 0: mul9 = 0; break;
	case 1: mul9 = a9; break;
	case 2: mul9 = 0; break;
	default: mul9 = 0; break;
	}
	switch(w10) {
	case -1: mul10 = -a10; break;
	case 0: mul10 = 0; break;
	case 1: mul10 = a10; break;
	case 2: mul10 = 0; break;
	default: mul10 = 0; break;
	}
	switch(w11) {
	case -1: mul11 = -a11; break;
	case 0: mul11 = 0; break;
	case 1: mul11 = a11; break;
	case 2: mul11 = 0; break;
	default: mul11 = 0; break;
	}
	switch(w12) {
	case -1: mul12 = -a12; break;
	case 0: mul12 = 0; break;
	case 1: mul12 = a12; break;
	case 2: mul12 = 0; break;
	default: mul12 = 0; break;
	}
	switch(w13) {
	case -1: mul13 = -a13; break;
	case 0: mul13 = 0; break;
	case 1: mul13 = a13; break;
	case 2: mul13 = 0; break;
	default: mul13 = 0; break;
	}
	switch(w14) {
	case -1: mul14 = -a14; break;
	case 0: mul14 = 0; break;
	case 1: mul14 = a14; break;
	case 2: mul14 = 0; break;
	default: mul14 = 0; break;
	}
	switch(w15) {
	case -1: mul15 = -a15; break;
	case 0: mul15 = 0; break;
	case 1: mul15 = a15; break;
	case 2: mul15 = 0; break;
	default: mul15 = 0; break;
	}

	add0_0 = mul0 + mul1;
	add0_1 = mul2 + mul3;
	add0_2 = mul4 + mul5;
	add0_3 = mul6 + mul7;
	add0_4 = mul8 + mul9;
	add0_5 = mul10 + mul11;
	add0_6 = mul12 + mul13;
	add0_7 = mul14 + mul15;

	add1_0 = add0_0 + add0_1;
	add1_1 = add0_2 + add0_3;
	add1_2 = add0_4 + add0_5;
	add1_3 = add0_6 + add0_7;

	add2_0 = add1_0 + add1_1;
	add2_1 = add1_2 + add1_3;

	add3_0 = add2_0 + add2_1;

	return add3_0;
}

