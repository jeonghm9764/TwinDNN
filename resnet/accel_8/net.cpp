#include "net.h"

void accel_8(
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
	FIX_W_8 weights[16][32];
#pragma HLS ARRAY_PARTITION variable=weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=weights complete dim=2
	FIX_W bias[16];
#pragma HLS ARRAY_PARTITION variable=bias cyclic factor=16 dim=1
	FIX_A_8 input_buf[32][64][64];
#pragma HLS ARRAY_PARTITION variable=input_buf complete dim=1
	FIX_ACC output_buf[16][64][64];
#pragma HLS ARRAY_PARTITION variable=output_buf complete dim=1

	FIX_256 temp_256;
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
	ap_uint<4> shift_w = SHIFT_W[layer];
	ap_uint<4> shift_a = SHIFT_A[layer];

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
								temp_act[cii] = pixel[cii];
								temp_act_8[cii] = temp_act[cii] << shift_a;
								input_buf[cii][h][w].range(BIT_A_8-1, 0) = temp_act_8[cii].range(BIT_A_8-1, 0);
							}
						}
					}

					for(ap_uint<7> h = 0; h < 29; h++) {
						for(ap_uint<7> w = 0; w < 29; w++) {
#pragma HLS PIPELINE
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								output_buf[coo][h][w] = 0;
							}
						}
					}

					for(ap_uint<3> i = 0; i < 7; i++) {
						for(ap_uint<3> j = 0; j < 7; j++) {
							for(ap_uint<5> wi = 0; wi < 16; wi++) {
#pragma HLS PIPELINE
								temp_256 = params[param_idx++];
								for(ap_uint<6> wj = 0; wj < 32; wj++) {
#pragma HLS UNROLL
									weights[wi][wj].range(BIT_W_8-1, 0) = temp_256.range(BIT_W_8*wj+BIT_W_8-1, BIT_W_8*wj);
								}
							}

							h_idx = i;
							for(ap_uint<7> h = 0; h < 28; h++, h_idx += 2) {
								w_idx = j;
								for(ap_uint<7> w = 0; w < 28; w++, w_idx += 2) {
#pragma HLS PIPELINE
									for(ap_uint<5> coo = 0; coo < 16; coo += 2) {
										compute_engine_32_8(temp_acc[coo], temp_acc[coo+1],
												weights[coo][0], weights[coo+1][0], input_buf[0][h_idx][w_idx],
												weights[coo][1], weights[coo+1][1], input_buf[1][h_idx][w_idx],
												weights[coo][2], weights[coo+1][2], input_buf[2][h_idx][w_idx],
												weights[coo][3], weights[coo+1][3], input_buf[3][h_idx][w_idx],
												weights[coo][4], weights[coo+1][4], input_buf[4][h_idx][w_idx],
												weights[coo][5], weights[coo+1][5], input_buf[5][h_idx][w_idx],
												weights[coo][6], weights[coo+1][6], input_buf[6][h_idx][w_idx],
												weights[coo][7], weights[coo+1][7], input_buf[7][h_idx][w_idx],
												weights[coo][8], weights[coo+1][8], input_buf[8][h_idx][w_idx],
												weights[coo][9], weights[coo+1][9], input_buf[9][h_idx][w_idx],
												weights[coo][10], weights[coo+1][10], input_buf[10][h_idx][w_idx],
												weights[coo][11], weights[coo+1][11], input_buf[11][h_idx][w_idx],
												weights[coo][12], weights[coo+1][12], input_buf[12][h_idx][w_idx],
												weights[coo][13], weights[coo+1][13], input_buf[13][h_idx][w_idx],
												weights[coo][14], weights[coo+1][14], input_buf[14][h_idx][w_idx],
												weights[coo][15], weights[coo+1][15], input_buf[15][h_idx][w_idx],
												weights[coo][16], weights[coo+1][16], input_buf[16][h_idx][w_idx],
												weights[coo][17], weights[coo+1][17], input_buf[17][h_idx][w_idx],
												weights[coo][18], weights[coo+1][18], input_buf[18][h_idx][w_idx],
												weights[coo][19], weights[coo+1][19], input_buf[19][h_idx][w_idx],
												weights[coo][20], weights[coo+1][20], input_buf[20][h_idx][w_idx],
												weights[coo][21], weights[coo+1][21], input_buf[21][h_idx][w_idx],
												weights[coo][22], weights[coo+1][22], input_buf[22][h_idx][w_idx],
												weights[coo][23], weights[coo+1][23], input_buf[23][h_idx][w_idx],
												weights[coo][24], weights[coo+1][24], input_buf[24][h_idx][w_idx],
												weights[coo][25], weights[coo+1][25], input_buf[25][h_idx][w_idx],
												weights[coo][26], weights[coo+1][26], input_buf[26][h_idx][w_idx],
												weights[coo][27], weights[coo+1][27], input_buf[27][h_idx][w_idx],
												weights[coo][28], weights[coo+1][28], input_buf[28][h_idx][w_idx],
												weights[coo][29], weights[coo+1][29], input_buf[29][h_idx][w_idx],
												weights[coo][30], weights[coo+1][30], input_buf[30][h_idx][w_idx],
												weights[coo][31], weights[coo+1][31], input_buf[31][h_idx][w_idx]);
										output_buf[coo][h][w] = output_buf[coo][h][w] + temp_acc[coo];
										output_buf[coo+1][h][w] = output_buf[coo+1][h][w] + temp_acc[coo+1];
									}
								}
							}
						}
					}

					for(ap_uint<7> h = 0; h < 28; h++) {
						for(ap_uint<7> w = 0; w < 28; w++) {
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS PIPELINE
								output_buf[coo][h][w] = (output_buf[coo][h][w] >> (shift_w + shift_a)) + bias[coo];
							}
						}
					}

					for(ap_uint<4> h = 0; h < 14; h++) {
						for(ap_uint<4> w = 0; w < 14; w++) {
#pragma HLS PIPELINE
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
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
									for(ap_uint<5> coo = 0; coo < 16; coo++) {
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
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								temp_act[coo] = output_buf[coo][h+32][w];
								temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo) = temp_act[coo].range(BIT_A-1, 0);
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

			for(ap_uint<7> h = 0; h < dim + 2; h++) {
				for(ap_uint<7> w = 0; w < dim + 2; w++) {
#pragma HLS PIPELINE
					for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
						output_buf[coo][h][w] = 0;
					}
				}
			}

			for(ap_uint<10> ci = 0; ci < cin; ci += 32) {
				for(ap_uint<7> h = 0; h < outdim; h++) {
					for(ap_uint<7> w = 0; w < outdim; w++) {
#pragma HLS PIPELINE
						temp_256 = input[ci>>4][h][w];
						for(ap_uint<6> cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
							temp_act[cii].range(BIT_A-1, 0) = temp_256.range(BIT_A*cii+BIT_A-1, BIT_A*cii);
						}
						temp_256 = input[(ci>>4)+1][h][w];
						for(ap_uint<6> cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
							temp_act[cii+16].range(BIT_A-1, 0) = temp_256.range(BIT_A*cii+BIT_A-1, BIT_A*cii);
						}
						for(ap_uint<6> cii = 0; cii < 32; cii++) {
#pragma HLS UNROLL
							temp_act_8[cii] = temp_act[cii] << shift_a;
							input_buf[cii][h][w].range(BIT_A_8-1, 0) = temp_act_8[cii].range(BIT_A_8-1, 0);
						}
					}
				}

				for(ap_uint<3> i = 0; i < kernel; i++) {
					for(ap_uint<3> j = 0; j < kernel; j++) {
						for(ap_uint<5> wi = 0; wi < 16; wi++) {
#pragma HLS PIPELINE
							temp_256 = params[param_idx++];
							for(ap_uint<6> wj = 0; wj < 32; wj++) {
#pragma HLS UNROLL
								weights[wi][wj].range(BIT_W_8-1, 0) = temp_256.range(BIT_W_8*wj+BIT_W_8-1, BIT_W_8*wj);
							}
						}

						h_idx = (kernel == 1) ? (ap_uint<8>)1 : (ap_uint<8>)i;
						for(ap_uint<7> h = 1; h < dim + 1; h++, h_idx += stride) {
							w_idx = (kernel == 1) ? (ap_uint<8>)1 : (ap_uint<8>)j;
							for(ap_uint<7> w = 1; w < dim + 1; w++, w_idx += stride) {
#pragma HLS PIPELINE
								for(ap_uint<5> coo = 0; coo < 16; coo += 2) {
#pragma HLS UNROLL
									compute_engine_32_8(temp_acc[coo], temp_acc[coo+1],
											weights[coo][0], weights[coo+1][0], input_buf[0][h_idx][w_idx],
											weights[coo][1], weights[coo+1][1], input_buf[1][h_idx][w_idx],
											weights[coo][2], weights[coo+1][2], input_buf[2][h_idx][w_idx],
											weights[coo][3], weights[coo+1][3], input_buf[3][h_idx][w_idx],
											weights[coo][4], weights[coo+1][4], input_buf[4][h_idx][w_idx],
											weights[coo][5], weights[coo+1][5], input_buf[5][h_idx][w_idx],
											weights[coo][6], weights[coo+1][6], input_buf[6][h_idx][w_idx],
											weights[coo][7], weights[coo+1][7], input_buf[7][h_idx][w_idx],
											weights[coo][8], weights[coo+1][8], input_buf[8][h_idx][w_idx],
											weights[coo][9], weights[coo+1][9], input_buf[9][h_idx][w_idx],
											weights[coo][10], weights[coo+1][10], input_buf[10][h_idx][w_idx],
											weights[coo][11], weights[coo+1][11], input_buf[11][h_idx][w_idx],
											weights[coo][12], weights[coo+1][12], input_buf[12][h_idx][w_idx],
											weights[coo][13], weights[coo+1][13], input_buf[13][h_idx][w_idx],
											weights[coo][14], weights[coo+1][14], input_buf[14][h_idx][w_idx],
											weights[coo][15], weights[coo+1][15], input_buf[15][h_idx][w_idx],
											weights[coo][16], weights[coo+1][16], input_buf[16][h_idx][w_idx],
											weights[coo][17], weights[coo+1][17], input_buf[17][h_idx][w_idx],
											weights[coo][18], weights[coo+1][18], input_buf[18][h_idx][w_idx],
											weights[coo][19], weights[coo+1][19], input_buf[19][h_idx][w_idx],
											weights[coo][20], weights[coo+1][20], input_buf[20][h_idx][w_idx],
											weights[coo][21], weights[coo+1][21], input_buf[21][h_idx][w_idx],
											weights[coo][22], weights[coo+1][22], input_buf[22][h_idx][w_idx],
											weights[coo][23], weights[coo+1][23], input_buf[23][h_idx][w_idx],
											weights[coo][24], weights[coo+1][24], input_buf[24][h_idx][w_idx],
											weights[coo][25], weights[coo+1][25], input_buf[25][h_idx][w_idx],
											weights[coo][26], weights[coo+1][26], input_buf[26][h_idx][w_idx],
											weights[coo][27], weights[coo+1][27], input_buf[27][h_idx][w_idx],
											weights[coo][28], weights[coo+1][28], input_buf[28][h_idx][w_idx],
											weights[coo][29], weights[coo+1][29], input_buf[29][h_idx][w_idx],
											weights[coo][30], weights[coo+1][30], input_buf[30][h_idx][w_idx],
											weights[coo][31], weights[coo+1][31], input_buf[31][h_idx][w_idx]);
									output_buf[coo][h][w] = output_buf[coo][h][w] + temp_acc[coo];
									output_buf[coo+1][h][w] = output_buf[coo+1][h][w] + temp_acc[coo+1];
								}
							}
						}
					}
				}
			}

			for(ap_uint<7> h = 1; h < dim + 1; h++) {
				for(ap_uint<7> w = 1; w < dim + 1; w++) {
#pragma HLS PIPELINE
					for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
						output_buf[coo][h][w] = (output_buf[coo][h][w] >> (shift_w + shift_a)) + bias[coo];
					}
				}
			}

			if(add == 1) {
				if(layer == NUM_LAYERS - 2) {
					for(ap_uint<7> h = 1; h < dim + 1; h++) {
						for(ap_uint<7> w = 1; w < dim + 1; w++) {
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
#pragma HLS PIPELINE
								output_buf[coo][h][w] = output_buf[coo][h][w] + temp_acc[coo];
							}
						}
					}
				}
				else {
					for(ap_uint<7> h = 1; h < dim + 1; h++) {
						for(ap_uint<7> w = 1; w < dim + 1; w++) {
							temp_256 = accum[co>>4][h][w];
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
								temp_act[coo].range(BIT_A-1, 0) = temp_256.range(BIT_A*coo+BIT_A-1, BIT_A*coo);
							}
							for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS PIPELINE
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
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				output_buf[coo][co>>5][co%32] = 0;
			}
		}

		for(ap_uint<10> ci = 0; ci < 512; ci += 32) {
			for(ap_uint<6> cii = 0; cii < 32; cii++) {
#pragma HLS UNROLL
				temp_acc[cii] = 0;
			}
			for(ap_uint<7> h = 1; h < 8; h++) {
				for(ap_uint<7> w = 1; w < 8; w++) {
#pragma HLS PIPELINE
					temp_256 = input[ci>>4][h][w];
					for(ap_uint<6> cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
						temp_act[cii].range(BIT_A-1, 0) = temp_256.range(BIT_A*cii+BIT_A-1, BIT_A*cii);
					}
					temp_256 = input[(ci>>4)+1][h][w];
					for(ap_uint<6> cii = 0; cii < 16; cii++) {
#pragma HLS UNROLL
						temp_act[cii+16].range(BIT_A-1, 0) = temp_256.range(BIT_A*cii+BIT_A-1, BIT_A*cii);
					}
					for(ap_uint<6> cii = 0; cii < 32; cii++) {
#pragma HLS PIPELINE
						temp_acc[cii] = temp_acc[cii] + temp_act[cii];
					}
				}
			}
			for(ap_uint<6> cii = 0; cii < 32; cii++) {
#pragma HLS PIPELINE
				temp_act[cii] = temp_acc[cii];
				temp_act_8[cii] = temp_act[cii] << shift_a;
			}

			for(ap_uint<10> co = 0; co < 1000; co += 16) {
				for(ap_uint<5> wi = 0; wi < 16; wi++) {
#pragma HLS PIPELINE
					temp_256 = params[param_idx++];
					for(ap_uint<6> wj = 0; wj < 32; wj++) {
#pragma HLS UNROLL
						weights[wi][wj].range(BIT_W_8-1, 0) = temp_256.range(BIT_W_8*wj+BIT_W_8-1, BIT_W_8*wj);
					}
				}
				for(ap_uint<5> coo = 0; coo < 16; coo += 2) {
#pragma HLS UNROLL
					compute_engine_32_8(temp_acc[coo], temp_acc[coo+1],
							weights[coo][0], weights[coo+1][0], temp_act_8[0],
							weights[coo][1], weights[coo+1][1], temp_act_8[1],
							weights[coo][2], weights[coo+1][2], temp_act_8[2],
							weights[coo][3], weights[coo+1][3], temp_act_8[3],
							weights[coo][4], weights[coo+1][4], temp_act_8[4],
							weights[coo][5], weights[coo+1][5], temp_act_8[5],
							weights[coo][6], weights[coo+1][6], temp_act_8[6],
							weights[coo][7], weights[coo+1][7], temp_act_8[7],
							weights[coo][8], weights[coo+1][8], temp_act_8[8],
							weights[coo][9], weights[coo+1][9], temp_act_8[9],
							weights[coo][10], weights[coo+1][10], temp_act_8[10],
							weights[coo][11], weights[coo+1][11], temp_act_8[11],
							weights[coo][12], weights[coo+1][12], temp_act_8[12],
							weights[coo][13], weights[coo+1][13], temp_act_8[13],
							weights[coo][14], weights[coo+1][14], temp_act_8[14],
							weights[coo][15], weights[coo+1][15], temp_act_8[15],
							weights[coo][16], weights[coo+1][16], temp_act_8[16],
							weights[coo][17], weights[coo+1][17], temp_act_8[17],
							weights[coo][18], weights[coo+1][18], temp_act_8[18],
							weights[coo][19], weights[coo+1][19], temp_act_8[19],
							weights[coo][20], weights[coo+1][20], temp_act_8[20],
							weights[coo][21], weights[coo+1][21], temp_act_8[21],
							weights[coo][22], weights[coo+1][22], temp_act_8[22],
							weights[coo][23], weights[coo+1][23], temp_act_8[23],
							weights[coo][24], weights[coo+1][24], temp_act_8[24],
							weights[coo][25], weights[coo+1][25], temp_act_8[25],
							weights[coo][26], weights[coo+1][26], temp_act_8[26],
							weights[coo][27], weights[coo+1][27], temp_act_8[27],
							weights[coo][28], weights[coo+1][28], temp_act_8[28],
							weights[coo][29], weights[coo+1][29], temp_act_8[29],
							weights[coo][30], weights[coo+1][30], temp_act_8[30],
							weights[coo][31], weights[coo+1][31], temp_act_8[31]);
					output_buf[coo][co>>5][co%32] = output_buf[coo][co>>5][co%32] + temp_acc[coo];
					output_buf[coo+1][co>>5][co%32] = output_buf[coo+1][co>>5][co%32] + temp_acc[coo+1];
				}
			}
		}

		for(ap_uint<10> co = 0; co < 1000; co += 16) {
#pragma HLS PIPELINE
			temp_256 = params[param_idx++];
			for(ap_uint<5> coo = 0; coo < 16; coo++) {
#pragma HLS UNROLL
				bias[coo].range(BIT_W-1, 0) = temp_256.range(BIT_W*coo+BIT_W-1, BIT_W*coo);
				output_buf[coo][co>>5][co%32] = (output_buf[coo][co>>5][co%32] >> (shift_w + shift_a)) + bias[coo];
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
		FIX_W_8 w0_31, FIX_W_8 w1_31, FIX_A_8 a31) {
	FIX_MUL mul0_0, mul0_1, mul0_2, mul0_3, mul0_4, mul0_5, mul0_6, mul0_7;
	FIX_MUL mul0_8, mul0_9, mul0_10, mul0_11, mul0_12, mul0_13, mul0_14, mul0_15;
	FIX_MUL mul0_16, mul0_17, mul0_18, mul0_19, mul0_20, mul0_21, mul0_22, mul0_23;
	FIX_MUL mul0_24, mul0_25, mul0_26, mul0_27, mul0_28, mul0_29, mul0_30, mul0_31;
	FIX_MUL mul1_0, mul1_1, mul1_2, mul1_3, mul1_4, mul1_5, mul1_6, mul1_7;
	FIX_MUL mul1_8, mul1_9, mul1_10, mul1_11, mul1_12, mul1_13, mul1_14, mul1_15;
	FIX_MUL mul1_16, mul1_17, mul1_18, mul1_19, mul1_20, mul1_21, mul1_22, mul1_23;
	FIX_MUL mul1_24, mul1_25, mul1_26, mul1_27, mul1_28, mul1_29, mul1_30, mul1_31;

	FIX_32 add0_0_0, add0_0_1, add0_0_2, add0_0_3, add0_0_4, add0_0_5, add0_0_6, add0_0_7;
	FIX_32 add0_0_8, add0_0_9, add0_0_10, add0_0_11, add0_0_12, add0_0_13, add0_0_14, add0_0_15;
	FIX_32 add1_0_0, add1_0_1, add1_0_2, add1_0_3, add1_0_4, add1_0_5, add1_0_6, add1_0_7;
	FIX_32 add1_0_8, add1_0_9, add1_0_10, add1_0_11, add1_0_12, add1_0_13, add1_0_14, add1_0_15;

	FIX_32 add0_1_0, add0_1_1, add0_1_2, add0_1_3, add0_1_4, add0_1_5, add0_1_6, add0_1_7;
	FIX_32 add1_1_0, add1_1_1, add1_1_2, add1_1_3, add1_1_4, add1_1_5, add1_1_6, add1_1_7;

	FIX_32 add0_2_0, add0_2_1, add0_2_2, add0_2_3;
	FIX_32 add1_2_0, add1_2_1, add1_2_2, add1_2_3;

	FIX_32 add0_3_0, add0_3_1;
	FIX_32 add1_3_0, add1_3_1;

	multiply(w0_0, w1_0, a0, mul0_0, mul1_0);
	multiply(w0_1, w1_1, a1, mul0_1, mul1_1);
	multiply(w0_2, w1_2, a2, mul0_2, mul1_2);
	multiply(w0_3, w1_3, a3, mul0_3, mul1_3);
	multiply(w0_4, w1_4, a4, mul0_4, mul1_4);
	multiply(w0_5, w1_5, a5, mul0_5, mul1_5);
	multiply(w0_6, w1_6, a6, mul0_6, mul1_6);
	multiply(w0_7, w1_7, a7, mul0_7, mul1_7);
	multiply(w0_8, w1_8, a8, mul0_8, mul1_8);
	multiply(w0_9, w1_9, a9, mul0_9, mul1_9);
	multiply(w0_10, w1_10, a10, mul0_10, mul1_10);
	multiply(w0_11, w1_11, a11, mul0_11, mul1_11);
	multiply(w0_12, w1_12, a12, mul0_12, mul1_12);
	multiply(w0_13, w1_13, a13, mul0_13, mul1_13);
	multiply(w0_14, w1_14, a14, mul0_14, mul1_14);
	multiply(w0_15, w1_15, a15, mul0_15, mul1_15);
	multiply(w0_16, w1_16, a16, mul0_16, mul1_16);
	multiply(w0_17, w1_17, a17, mul0_17, mul1_17);
	multiply(w0_18, w1_18, a18, mul0_18, mul1_18);
	multiply(w0_19, w1_19, a19, mul0_19, mul1_19);
	multiply(w0_20, w1_20, a20, mul0_20, mul1_20);
	multiply(w0_21, w1_21, a21, mul0_21, mul1_21);
	multiply(w0_22, w1_22, a22, mul0_22, mul1_22);
	multiply(w0_23, w1_23, a23, mul0_23, mul1_23);
	multiply(w0_24, w1_24, a24, mul0_24, mul1_24);
	multiply(w0_25, w1_25, a25, mul0_25, mul1_25);
	multiply(w0_26, w1_26, a26, mul0_26, mul1_26);
	multiply(w0_27, w1_27, a27, mul0_27, mul1_27);
	multiply(w0_28, w1_28, a28, mul0_28, mul1_28);
	multiply(w0_29, w1_29, a29, mul0_29, mul1_29);
	multiply(w0_30, w1_30, a30, mul0_30, mul1_30);
	multiply(w0_31, w1_31, a31, mul0_31, mul1_31);

	add0_0_0 = mul0_0 + mul0_1;
	add0_0_1 = mul0_2 + mul0_3;
	add0_0_2 = mul0_4 + mul0_5;
	add0_0_3 = mul0_6 + mul0_7;
	add0_0_4 = mul0_8 + mul0_9;
	add0_0_5 = mul0_10 + mul0_11;
	add0_0_6 = mul0_12 + mul0_13;
	add0_0_7 = mul0_14 + mul0_15;
	add0_0_8 = mul0_16 + mul0_17;
	add0_0_9 = mul0_18 + mul0_19;
	add0_0_10 = mul0_20 + mul0_21;
	add0_0_11 = mul0_22 + mul0_23;
	add0_0_12 = mul0_24 + mul0_25;
	add0_0_13 = mul0_26 + mul0_27;
	add0_0_14 = mul0_28 + mul0_29;
	add0_0_15 = mul0_30 + mul0_31;

	add1_0_0 = mul1_0 + mul1_1;
	add1_0_1 = mul1_2 + mul1_3;
	add1_0_2 = mul1_4 + mul1_5;
	add1_0_3 = mul1_6 + mul1_7;
	add1_0_4 = mul1_8 + mul1_9;
	add1_0_5 = mul1_10 + mul1_11;
	add1_0_6 = mul1_12 + mul1_13;
	add1_0_7 = mul1_14 + mul1_15;
	add1_0_8 = mul1_16 + mul1_17;
	add1_0_9 = mul1_18 + mul1_19;
	add1_0_10 = mul1_20 + mul1_21;
	add1_0_11 = mul1_22 + mul1_23;
	add1_0_12 = mul1_24 + mul1_25;
	add1_0_13 = mul1_26 + mul1_27;
	add1_0_14 = mul1_28 + mul1_29;
	add1_0_15 = mul1_30 + mul1_31;

	add0_1_0 = add0_0_0 + add0_0_1;
	add0_1_1 = add0_0_2 + add0_0_3;
	add0_1_2 = add0_0_4 + add0_0_5;
	add0_1_3 = add0_0_6 + add0_0_7;
	add0_1_4 = add0_0_8 + add0_0_9;
	add0_1_5 = add0_0_10 + add0_0_11;
	add0_1_6 = add0_0_12 + add0_0_13;
	add0_1_7 = add0_0_14 + add0_0_15;

	add1_1_0 = add1_0_0 + add1_0_1;
	add1_1_1 = add1_0_2 + add1_0_3;
	add1_1_2 = add1_0_4 + add1_0_5;
	add1_1_3 = add1_0_6 + add1_0_7;
	add1_1_4 = add1_0_8 + add1_0_9;
	add1_1_5 = add1_0_10 + add1_0_11;
	add1_1_6 = add1_0_12 + add1_0_13;
	add1_1_7 = add1_0_14 + add1_0_15;

	add0_2_0 = add0_1_0 + add0_1_1;
	add0_2_1 = add0_1_2 + add0_1_3;
	add0_2_2 = add0_1_4 + add0_1_5;
	add0_2_3 = add0_1_6 + add0_1_7;

	add1_2_0 = add1_1_0 + add1_1_1;
	add1_2_1 = add1_1_2 + add1_1_3;
	add1_2_2 = add1_1_4 + add1_1_5;
	add1_2_3 = add1_1_6 + add1_1_7;

	add0_3_0 = add0_2_0 + add0_2_1;
	add0_3_1 = add0_2_2 + add0_2_3;

	add1_3_0 = add1_2_0 + add1_2_1;
	add1_3_1 = add1_2_2 + add1_2_3;

	o0 = add0_3_0 + add0_3_1;
	o1 = add1_3_0 + add1_3_1;
}

void multiply(FIX_W_8 w1, FIX_W_8 w2, FIX_A_8 a, FIX_MUL &m1, FIX_MUL &m2) {
	ap_int<26> ai = 0;
	ap_int<8> bi, ci;
	ap_int<45> rst;
#pragma HLS RESOURCE variable=rst core=DSP48
	ap_uint<19> cst;

	ai.range(7, 0) = w1.range(7, 0);
	bi.range(7, 0) = w2.range(7, 0);
	ci.range(7, 0) = a.range(7, 0);

	if(((ci < 0) && (bi > 0)) || ((ci > 0) && (bi < 0))) cst = 0x40000;
	else cst = 0;

	rst = ((ai << 18) + bi) * ci + cst;

	m1.range(15, 0) = rst.range(33, 18);
	m2.range(15, 0) = rst.range(15, 0);
}
