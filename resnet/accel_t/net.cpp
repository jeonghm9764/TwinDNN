#include "net.h"

void accel_t(
		UFIX_32 image[230][230],
		FIX_256 buffers[3][32][64][64],
		FIX_256 params[NUM_PARAMS],
		float probs[1000]) {
#pragma HLS INTERFACE m_axi port=image offset=slave bundle=IMAGE
#pragma HLS INTERFACE m_axi port=params offset=slave bundle=PARAM
#pragma HLS INTERFACE m_axi port=buffers offset=slave bundle=BUFFER
#pragma HLS INTERFACE m_axi port=probs offset=slave bundle=PROB
#pragma HLS INTERFACE s_axilite port=return
	for(int l = 0; l < NUM_LAYERS; l++) {
		layer(image, buffers[INPUT[l]], params, buffers[ACCUM[l]], buffers[OUTPUT[l]], probs, l);
	}
}

void layer(
		UFIX_32 image[230][230],
		FIX_256 input[32][64][64],
		FIX_256 params[NUM_PARAMS],
		FIX_256 accum[32][64][64],
		FIX_256 output[32][64][64],
		float probs[1000],
		int layer_num) {
	FIX_2 weights_t[32][32];
#pragma HLS ARRAY_PARTITION variable=weights_t complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights_t complete dim=2
	FIX_W weights[16][16];
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
	FIX_W weights_scale[32];
#pragma HLS ARRAY_PARTITION variable=weights_scale complete dim=1
	FIX_W bias[32];
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1
	FIX_A_8 input_buf[16][64][64];
#pragma HLS ARRAY_PARTITION variable=input_buf complete dim=1
	FIX_ACC output_buf[32][64][64];
#pragma HLS ARRAY_PARTITION variable=output_buf complete dim=1

	FIX_256 temp_256;
	FIX_64 temp_64;
	UFIX_32 temp_32;
	FIX_A temp_act[32];
#pragma HLS ARRAY_PARTITION variable=temp_act complete dim=1
	FIX_A_8 temp_act_8[32];
#pragma HLS ARRAY_PARTITION variable=temp_act_8 complete dim=1
	FIX_ACC temp_acc[32];
#pragma HLS ARRAY_PARTITION variable=temp_acc complete dim=1
	ap_fixed<9, 1, AP_RND_ZERO, AP_SAT> pixel[3];
#pragma HLS ARRAY_PARTITION variable=pixel complete dim=1

	ap_uint<5> layer = layer_num;
	ap_uint<10> cin = CIN[layer];
	ap_uint<10> cout = COUT[layer];
	ap_uint<7> dim = DIM[layer];
	ap_uint<3> kernel = KERNEL[layer];
	int param_idx = IDX[layer];
	ap_uint<2> stride = (cin == cout) ? 1 : 2;
	ap_uint<7> outdim = dim * stride + 2;
	bool relu = RELU[layer];
	bool add = ADD[layer];
	ap_uint<8> h_idx, w_idx;
	ap_uint<4> shift_a = SHIFT_A[layer];

	// conv1 layer
	if(layer == 0) {
		for(ap_uint<10> co = 0; co < 64; co += 32) {
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo+16].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				weights_scale[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				weights_scale[coo+16].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}

			for(ap_uint<8> h_in_mul = 0, h_out_mul = 0; h_in_mul < 224; h_in_mul += 56, h_out_mul += 14) {
				for(ap_uint<8> w_in_mul = 0, w_out_mul = 0; w_in_mul < 224; w_in_mul += 56, w_out_mul += 14) {
					// load image
					for(ap_uint<6> h = 0; h < 62; h++) {
						for(ap_uint<6> w = 0; w < 62; w++) {
#pragma HLS PIPELINE
							temp_32 = image[h_in_mul+h][w_in_mul+w];
							for(ap_uint<6> cii = 0; cii < 3; cii++) {
#pragma HLS UNROLL
								pixel[cii].range(8, 0) = temp_32.range(9*cii+8, 9*cii);
								temp_act[cii] = pixel[cii];
								temp_act_8[cii] = temp_act[cii] << shift_a;
								input_buf[cii][h][w].range(BIT_A_8-1, 0) = temp_act_8[cii].range(BIT_A_8-1, 0);
							}
						}
					}

					for(ap_uint<7> h = 0; h < 29; h++) {
						for(ap_uint<7> w = 0; w < 29; w++) {
#pragma HLS PIPELINE
							for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
								output_buf[coo][h][w] = 0;
							}
						}
					}

					for(ap_uint<3> i = 0; i < 7; i++) {
						for(ap_uint<3> j = 0; j < 7; j++) {
							for(ap_uint<6> wi = 0; wi < 32; wi++) {
#pragma HLS PIPELINE
								temp_256 = params[param_idx++];
								temp_64.range(63, 0) = temp_256.range(63, 0);
								for(ap_uint<6> wj = 0; wj < 32; wj++) {
#pragma HLS UNROLL
									weights_t[wi][wj].range(1, 0) = temp_64.range(wj*2+1, wj*2);
								}
							}

							h_idx = i;
							for(ap_uint<7> h = 0; h < 28; h++, h_idx += 2) {
								w_idx = j;
								for(ap_uint<7> w = 0; w < 28; w++, w_idx += 2) {
#pragma HLS PIPELINE
									for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
										output_buf[coo][h][w] = output_buf[coo][h][w] +
												compute_engine_32_t(weights_t[coo][0], input_buf[0][h_idx][w_idx],
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

					for(ap_uint<7> h = 0; h < 28; h++) {
						for(ap_uint<7> w = 0; w < 28; w++) {
							for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
								output_buf[coo][h][w] = output_buf[coo][h][w] >> shift_a;
							}
							for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS PIPELINE
								output_buf[coo][h][w] = output_buf[coo][h][w] * weights_scale[coo] + bias[coo];
							}
						}
					}

					for(ap_uint<4> h = 0; h < 14; h++) {
						for(ap_uint<4> w = 0; w < 14; w++) {
#pragma HLS PIPELINE
							for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
								output_buf[coo][h+32][w] = 0;
							}
						}
					}

					for(ap_uint<3> i = 0; i < 3; i++) {
						for(ap_uint<3> j = 0; j < 3; j++) {
							h_idx = i;
							for(ap_uint<7> h = 0; h < 14; h++, h_idx += 2) {
								w_idx = j;
								for(ap_uint<7> w = 0; w < 14; w++, w_idx += 2) {
#pragma HLS PIPELINE
									for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
										if(output_buf[coo][h+32][w] < output_buf[coo][h_idx][w_idx])
											output_buf[coo][h+32][w] = output_buf[coo][h_idx][w_idx];
									}
								}
							}
						}
					}

					for(ap_uint<7> h = 0; h < 14; h++) {
						for(ap_uint<7> w = 0; w < 14; w++) {
#pragma HLS PIPELINE
							for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
								temp_act[coo] = output_buf[coo][h+32][w];
							}

							for(ap_uint<6> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo) = temp_act[coo].range(BIT_A-1, 0);
							}
							output[co>>4][h+h_out_mul+1][w+w_out_mul+1] = temp_256;
							for(ap_uint<6> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo) = temp_act[coo+16].range(BIT_A-1, 0);
							}
							output[(co>>4)+1][h+h_out_mul+1][w+w_out_mul+1] = temp_256;
						}
					}
				}
			}
		}

		for(ap_uint<3> co = 0; co < 4; co++) {
			for(ap_uint<6> hw = 0; hw < 58; hw++) {
				output[co][0][hw] = 0;
				output[co][57][hw] = 0;
				output[co][hw][0] = 0;
				output[co][hw][57] = 0;
			}
		}
	}

	// other layers
	else if(layer < NUM_LAYERS-1) {
		for(ap_uint<10> co = 0; co < cout; co += 32) {
#pragma HLS loop_tripcount max=4
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo+16].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				weights_scale[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				weights_scale[coo+16].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}

			for(ap_uint<7> h = 0; h < dim + 2; h++) {
#pragma HLS loop_tripcount max=58
				for(ap_uint<7> w = 0; w < dim + 2; w++) {
#pragma HLS loop_tripcount max=58
#pragma HLS PIPELINE
					for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
						output_buf[coo][h][w] = 0;
					}
				}
			}

			if(co == 0) { // no function, do not remove
				output[layer_num][0][0] = 0;
			}

			for(ap_uint<10> ci = 0; ci < cin; ci += 16) {
#pragma HLS loop_tripcount max=2
				for(ap_uint<7> h = 0; h < outdim; h++) {
#pragma HLS loop_tripcount max=58
					for(ap_uint<7> w = 0; w < outdim; w++) {
#pragma HLS loop_tripcount max=58
#pragma HLS PIPELINE
						temp_256 = input[ci>>4][h][w];
						for(ap_uint<6> cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
							temp_act[cii].range(BIT_A-1, 0) = temp_256.range(BIT_A*cii+BIT_A-1, BIT_A*cii);
							temp_act_8[cii] = temp_act[cii] << shift_a;
							input_buf[cii][h][w].range(BIT_A_8-1, 0) = temp_act_8[cii].range(BIT_A_8-1, 0);
						}
					}
				}

				for(ap_uint<3> i = 0; i < kernel; i++) {
#pragma HLS loop_tripcount max=3
					for(ap_uint<3> j = 0; j < kernel; j++) {
#pragma HLS loop_tripcount max=3
						for(ap_uint<6> wi = 0; wi < 32; wi++) {
#pragma HLS PIPELINE
							temp_256 = params[param_idx++];
							temp_64.range(63, 0) = temp_256.range(63, 0);
							for(ap_uint<6> wj = 0; wj < 32; wj++) {
#pragma HLS UNROLL
								weights_t[wi][wj].range(1, 0) = temp_64.range(wj*2+1, wj*2);
							}
						}

						h_idx = (kernel == 1) ? (ap_uint<8>)1 : (ap_uint<8>)i;
						for(ap_uint<7> h = 1; h < dim + 1; h++, h_idx += stride) {
#pragma HLS loop_tripcount max=56
							w_idx = (kernel == 1) ? (ap_uint<8>)1 : (ap_uint<8>)j;
							for(ap_uint<7> w = 1; w < dim + 1; w++, w_idx += stride) {
#pragma HLS loop_tripcount max=56
#pragma HLS PIPELINE
								for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
									output_buf[coo][h][w] = output_buf[coo][h][w] +
											compute_engine_32_t(weights_t[coo][0], input_buf[0][h_idx][w_idx],
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

			for(ap_uint<7> h = 1; h < dim + 1; h++) {
#pragma HLS loop_tripcount max=56
				for(ap_uint<7> w = 1; w < dim + 1; w++) {
#pragma HLS loop_tripcount max=56
					for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
						output_buf[coo][h][w] = output_buf[coo][h][w] >> shift_a;
					}
					for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS PIPELINE
						output_buf[coo][h][w] = output_buf[coo][h][w] * weights_scale[coo] + bias[coo];
					}
				}
			}

			if(add == 1) {
				for(ap_uint<7> h = 1; h < dim + 1; h++) {
#pragma HLS loop_tripcount max=56
					for(ap_uint<7> w = 1; w < dim + 1; w++) {
#pragma HLS loop_tripcount max=56
#pragma HLS PIPELINE
						temp_256 = accum[co>>4][h][w];
						for(ap_uint<6> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
							temp_act[coo].range(BIT_A-1, 0) = temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo);
						}
						temp_256 = accum[(co>>4)+1][h][w];
						for(ap_uint<6> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
							temp_act[coo+16].range(BIT_A-1, 0) = temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo);
						}
						for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
							output_buf[coo][h][w] = output_buf[coo][h][w] + temp_act[coo];
						}
					}
				}
			}

			if(relu == 1) {
				for(ap_uint<7> h = 1; h < dim + 1; h++) {
#pragma HLS loop_tripcount max=56
					for(ap_uint<7> w = 1; w < dim + 1; w++) {
#pragma HLS loop_tripcount max=56
#pragma HLS PIPELINE
						for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
							if(output_buf[coo][h][w] < 0)
								output_buf[coo][h][w] = 0;
						}
					}
				}
			}

			for(ap_uint<7> h = 0; h < dim + 2; h++) {
#pragma HLS loop_tripcount max=58
				for(ap_uint<7> w = 0; w < dim + 2; w++) {
#pragma HLS loop_tripcount max=58
#pragma HLS PIPELINE
					for(ap_uint<6> coo = 0; coo < 32; coo++) {
#pragma HLS UNROLL
						temp_act[coo] = output_buf[coo][h][w];
					}

					for(ap_uint<6> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
						temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo) = temp_act[coo].range(BIT_A-1, 0);
					}
					output[co>>4][h][w] = temp_256;
					for(ap_uint<6> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
						temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo) = temp_act[coo+16].range(BIT_A-1, 0);
					}
					output[(co>>4)+1][h][w] = temp_256;
				}
			}
		}
	}

	// fc layer
	else {
		for(ap_uint<10> co = 0; co < 1000; co += 16) {
#pragma HLS PIPELINE
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
				output_buf[coo][co>>5][co%32] = bias[coo];
			}
		}

		for(ap_uint<10> ci = 0; ci < 512; ci += 16) {
			for(ap_uint<5> cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
				temp_acc[cii] = 0;
			}
			for(ap_uint<7> h = 1; h < 8; h++) {
				for(ap_uint<7> w = 1; w < 8; w++) {
#pragma HLS PIPELINE
					temp_256 = input[ci>>4][h][w];
					for(ap_uint<5> cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
						temp_act[cii].range(BIT_A-1, 0) = temp_256.range(BIT_A*cii+BIT_A-1, BIT_A*cii);
						temp_acc[cii] = temp_acc[cii] + temp_act[cii];
					}
				}
			}
			for(ap_uint<5> cii = 0; cii < 16; cii++) {
#pragma HLS PIPELINE
				temp_act[cii] = temp_acc[cii] / 49;
			}

			for(ap_uint<10> co = 0; co < 1000; co += 16) {
				for(ap_uint<5> wi = 0; wi < 16; wi++) {
#pragma HLS PIPELINE
					temp_256 = params[param_idx++];
					for(ap_uint<5> wj = 0; wj < 16; wj++) {
#pragma HLS UNROLL
						weights[wi][wj].range(BIT_W-1, 0) = temp_256.range(BIT_W*wj+BIT_W-1, BIT_W*wj);
					}
				}
				for(ap_uint<5> cii = 0; cii < 16; cii++) {
					for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS PIPELINE
						output_buf[coo][co>>5][co%32] = output_buf[coo][co>>5][co%32] + weights[coo][cii] * temp_act[cii];
					}
				}
			}
		}

		for(ap_uint<10> co = 0; co < 1000; co += 16) {
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS PIPELINE
				if(co+coo < 1000) {
					probs[co+coo] = output_buf[coo][co>>5][co%32];
				}
			}
		}
	}
	//printf("%d\n", param_idx);
}

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
		FIX_2 w15, FIX_A_8 a15) {
	ap_fixed<BIT_A_8, 6, AP_RND_ZERO, AP_SAT> mul0, mul1, mul2, mul3, mul4, mul5, mul6, mul7;
	ap_fixed<BIT_A_8, 6, AP_RND_ZERO, AP_SAT> mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	ap_fixed<BIT_A_8+1, 7, AP_RND_ZERO, AP_SAT> add0_0, add0_1, add0_2, add0_3, add0_4, add0_5, add0_6, add0_7;
	ap_fixed<BIT_A_8+2, 8, AP_RND_ZERO, AP_SAT> add1_0, add1_1, add1_2, add1_3;
	ap_fixed<BIT_A_8+3, 9, AP_RND_ZERO, AP_SAT> add2_0, add2_1;
	FIX_ACC add3_0;

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

