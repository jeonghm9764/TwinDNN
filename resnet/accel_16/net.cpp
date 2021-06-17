#include "net.h"

void accel_16(
		UFIX_32 image[230][230],
		FIX_256 buffers[3][32][64][64],
		FIX_256 params[NUM_PARAMS],
		float probs[1000]
		) {
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
	FIX_W weights[16][16];
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
	FIX_W bias[16];
#pragma HLS ARRAY_PARTITION variable=bias complete dim=1
	FIX_A input_buf[16][64][64];
#pragma HLS ARRAY_PARTITION variable=input_buf complete dim=1
	FIX_ACC output_buf[16][64][64];
#pragma HLS ARRAY_PARTITION variable=output_buf complete dim=1

	FIX_256 temp_256;
	UFIX_32 temp_32;
	FIX_A temp_act[16];
#pragma HLS ARRAY_PARTITION variable=temp_act complete dim=1
	FIX_ACC temp_acc[16];
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

	// conv1 layer
	if(layer == 0) {
		for(ap_uint<10> co = 0; co < 64; co += 16) {
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}

			for(ap_uint<8> h_in_mul = 0, h_out_mul = 0; h_in_mul < 224; h_in_mul += 56, h_out_mul += 14) {
				for(ap_uint<8> w_in_mul = 0, w_out_mul = 0; w_in_mul < 224; w_in_mul += 56, w_out_mul += 14) {
					// load image
					for(ap_uint<6> h = 0; h < 62; h++) {
						for(ap_uint<6> w = 0; w < 62; w++) {
#pragma HLS PIPELINE
							temp_32 = image[h_in_mul+h][w_in_mul+w];
							for(ap_uint<5> cii = 0; cii < 3; cii++) {
#pragma HLS UNROLL
								pixel[cii].range(8, 0) = temp_32.range(9*cii+8, 9*cii);
							}
							input_buf[0][h][w] = pixel[0];
							input_buf[1][h][w] = pixel[1];
							input_buf[2][h][w] = pixel[2];
						}
					}

					for(ap_uint<7> h = 0; h < 28; h++) {
						for(ap_uint<7> w = 0; w < 28; w++) {
#pragma HLS PIPELINE
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								output_buf[coo][h][w] = bias[coo];
							}
						}
					}

					for(ap_uint<7> hw = 0; hw < 29; hw++) {
#pragma HLS PIPELINE
						for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
							output_buf[coo][hw][28] = 0;
							output_buf[coo][28][hw] = 0;
						}
					}


					for(ap_uint<3> i = 0; i < 7; i++) {
						for(ap_uint<3> j = 0; j < 7; j++) {
							for(ap_uint<5> wi = 0; wi < 16; wi++) {
#pragma HLS PIPELINE
								temp_256 = params[param_idx++];
								for(ap_uint<5> wj = 0; wj < 16; wj++) {
#pragma HLS UNROLL
									weights[wi][wj].range(BIT_W-1, 0) = temp_256.range(BIT_W*wj+BIT_W-1, BIT_W*wj);
								}
							}

							h_idx = i;
							for(ap_uint<7> h = 0; h < 28; h++, h_idx += 2) {
								w_idx = j;
								for(ap_uint<7> w = 0; w < 28; w++, w_idx += 2) {
#pragma HLS PIPELINE
									for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
										output_buf[coo][h][w] = output_buf[coo][h][w] + compute_engine_16(
												weights[coo][0],   input_buf[0][h_idx][w_idx],
												weights[coo][1],   input_buf[1][h_idx][w_idx],
												weights[coo][2],   input_buf[2][h_idx][w_idx],
												weights[coo][3],   input_buf[3][h_idx][w_idx],
												weights[coo][4],   input_buf[4][h_idx][w_idx],
												weights[coo][5],   input_buf[5][h_idx][w_idx],
												weights[coo][6],   input_buf[6][h_idx][w_idx],
												weights[coo][7],   input_buf[7][h_idx][w_idx],
												weights[coo][8],   input_buf[8][h_idx][w_idx],
												weights[coo][9],   input_buf[9][h_idx][w_idx],
												weights[coo][10],  input_buf[10][h_idx][w_idx],
												weights[coo][11],  input_buf[11][h_idx][w_idx],
												weights[coo][12],  input_buf[12][h_idx][w_idx],
												weights[coo][13],  input_buf[13][h_idx][w_idx],
												weights[coo][14],  input_buf[14][h_idx][w_idx],
												weights[coo][15],  input_buf[15][h_idx][w_idx]);
									}
								}
							}
						}
					}

					for(ap_uint<4> h = 0; h < 14; h++) {
						for(ap_uint<4> w = 0; w < 14; w++) {
#pragma HLS PIPELINE
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								input_buf[coo][h][w] = 0;
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
									for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
										if(input_buf[coo][h][w] < output_buf[coo][h_idx][w_idx])
											input_buf[coo][h][w] = output_buf[coo][h_idx][w_idx];
									}
								}
							}
						}
					}

					for(ap_uint<7> h = 0; h < 14; h++) {
						for(ap_uint<7> w = 0; w < 14; w++) {
#pragma HLS PIPELINE
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo) = input_buf[coo][h][w].range(BIT_A-1, 0);
							}
							output[co>>4][h+h_out_mul+1][w+w_out_mul+1] = temp_256;
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
		for(ap_uint<10> co = 0; co < cout; co += 16) {
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
			}

			for(ap_uint<7> h = 1; h < dim + 1; h++) {
				for(ap_uint<7> w = 1; w < dim + 1; w++) {
#pragma HLS PIPELINE
					for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
						output_buf[coo][h][w] = bias[coo];
					}
				}
			}

			for(ap_uint<10> ci = 0; ci < cin; ci += 16) {
				for(ap_uint<7> h = 0; h < outdim; h++) {
					for(ap_uint<7> w = 0; w < outdim; w++) {
#pragma HLS PIPELINE
						temp_256 = input[ci>>4][h][w];
						for(ap_uint<5> cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
							input_buf[cii][h][w].range(BIT_A-1, 0) = temp_256.range(BIT_A*cii+BIT_A-1, BIT_A*cii);
						}
					}
				}

				for(ap_uint<3> i = 0; i < kernel; i++) {
					for(ap_uint<3> j = 0; j < kernel; j++) {
						for(ap_uint<5> wi = 0; wi < 16; wi++) {
#pragma HLS PIPELINE
							temp_256 = params[param_idx++];
							for(ap_uint<5> wj = 0; wj < 16; wj++) {
#pragma HLS UNROLL
								weights[wi][wj].range(BIT_W-1, 0) = temp_256.range(BIT_W*wj+BIT_W-1, BIT_W*wj);
							}
						}

						h_idx = (kernel == 1) ? (ap_uint<8>)1 : (ap_uint<8>)i;
						for(ap_uint<7> h = 1; h < dim + 1; h++, h_idx += stride) {
							w_idx = (kernel == 1) ? (ap_uint<8>)1 : (ap_uint<8>)j;
							for(ap_uint<7> w = 1; w < dim + 1; w++, w_idx += stride) {
#pragma HLS PIPELINE
								for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
									output_buf[coo][h][w] = output_buf[coo][h][w] + compute_engine_16(
											weights[coo][0],   input_buf[0][h_idx][w_idx],
											weights[coo][1],   input_buf[1][h_idx][w_idx],
											weights[coo][2],   input_buf[2][h_idx][w_idx],
											weights[coo][3],   input_buf[3][h_idx][w_idx],
											weights[coo][4],   input_buf[4][h_idx][w_idx],
											weights[coo][5],   input_buf[5][h_idx][w_idx],
											weights[coo][6],   input_buf[6][h_idx][w_idx],
											weights[coo][7],   input_buf[7][h_idx][w_idx],
											weights[coo][8],   input_buf[8][h_idx][w_idx],
											weights[coo][9],   input_buf[9][h_idx][w_idx],
											weights[coo][10],  input_buf[10][h_idx][w_idx],
											weights[coo][11],  input_buf[11][h_idx][w_idx],
											weights[coo][12],  input_buf[12][h_idx][w_idx],
											weights[coo][13],  input_buf[13][h_idx][w_idx],
											weights[coo][14],  input_buf[14][h_idx][w_idx],
											weights[coo][15],  input_buf[15][h_idx][w_idx]);
								}
							}
						}
					}
				}
			}

			if(add == 1) {
				if(layer == NUM_LAYERS - 2) {
					for(ap_uint<7> h = 1; h < dim + 1; h++) {
						for(ap_uint<7> w = 1; w < dim + 1; w++) {
#pragma HLS PIPELINE
							temp_256 = accum[co>>4][h][w];
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								temp_act[coo].range(BIT_A-1, 0) = temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo);
							}
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								temp_acc[coo] = temp_act[coo] / 49;
							}
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								output_buf[coo][h][w] = output_buf[coo][h][w] + temp_acc[coo];
							}
						}
					}
				}
				else {
					for(ap_uint<7> h = 1; h < dim + 1; h++) {
						for(ap_uint<7> w = 1; w < dim + 1; w++) {
#pragma HLS PIPELINE
							temp_256 = accum[co>>4][h][w];
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								temp_act[coo].range(BIT_A-1, 0) = temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo);
							}
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								output_buf[coo][h][w] = output_buf[coo][h][w] + temp_act[coo];
							}
						}
					}
				}
			}

			for(ap_uint<7> hw = 0; hw < dim + 2; hw++) {
#pragma HLS PIPELINE ii=4
				for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
					output_buf[coo][0][hw] = 0;
					output_buf[coo][dim+1][hw] = 0;
					output_buf[coo][hw][0] = 0;
					output_buf[coo][hw][dim+1] = 0;
				}
			}

			if(relu == 1) {
				for(ap_uint<7> h = 1; h < dim + 1; h++) {
					for(ap_uint<7> w = 1; w < dim + 1; w++) {
#pragma HLS PIPELINE
						for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
							if(output_buf[coo][h][w] < 0)
								output_buf[coo][h][w] = 0;
						}
					}
				}
			}

			for(ap_uint<7> h = 0; h < dim + 2; h++) {
				for(ap_uint<7> w = 0; w < dim + 2; w++) {
#pragma HLS PIPELINE
					for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
						temp_act[coo] = output_buf[coo][h][w];
					}
					for(ap_uint<5> coo = 0; coo < 16; coo++) {
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
				temp_act[cii] = temp_acc[cii];
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

				for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
					output_buf[coo][co>>5][co%32] = output_buf[coo][co>>5][co%32] + compute_engine_16(
							weights[coo][0],   temp_act[0],
							weights[coo][1],   temp_act[1],
							weights[coo][2],   temp_act[2],
							weights[coo][3],   temp_act[3],
							weights[coo][4],   temp_act[4],
							weights[coo][5],   temp_act[5],
							weights[coo][6],   temp_act[6],
							weights[coo][7],   temp_act[7],
							weights[coo][8],   temp_act[8],
							weights[coo][9],   temp_act[9],
							weights[coo][10],  temp_act[10],
							weights[coo][11],  temp_act[11],
							weights[coo][12],  temp_act[12],
							weights[coo][13],  temp_act[13],
							weights[coo][14],  temp_act[14],
							weights[coo][15],  temp_act[15]);
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
					  FIX_W w15, FIX_A b15) {
	FIX_32 mul0, mul1, mul2,  mul3,  mul4,  mul5,  mul6,  mul7;
	FIX_32 mul8, mul9, mul10, mul11, mul12, mul13, mul14, mul15;
	FIX_32 add0, add1, add2, add3,  add4,  add5,  add6;
	FIX_32 add7, add8, add9, add10, add11, add12, add13, add14;

	mul0  = w0  * b0;
	mul1  = w1  * b1;
	mul2  = w2  * b2;
	mul3  = w3  * b3;
	mul4  = w4  * b4;
	mul5  = w5  * b5;
	mul6  = w6  * b6;
	mul7  = w7  * b7;
	mul8  = w8  * b8;
	mul9  = w9  * b9;
	mul10 = w10 * b10;
	mul11 = w11 * b11;
	mul12 = w12 * b12;
	mul13 = w13 * b13;
	mul14 = w14 * b14;
	mul15 = w15 * b15;

	add0 = mul0  + mul1;
	add1 = mul2  + mul3;
	add2 = mul4  + mul5;
	add3 = mul6  + mul7;
	add4 = mul8  + mul9;
	add5 = mul10 + mul11;
	add6 = mul12 + mul13;
	add7 = mul14 + mul15;

	add8  = add0 + add1;
	add9  = add2 + add3;
	add10 = add4 + add5;
	add11 = add6 + add7;

	add12 = add8  + add9;
	add13 = add10 + add11;

	add14 = add12 + add13;

	return add14;
}
