// AnyNNet.h
// >>Machine Learning in C without Matrices<<
// based on Tsoding Daily (https://www.youtube.com/watch?v=PGSba51aRYU&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw&index=1)
// process any fully connected system: arch{numInputs, numNodes of layer i, .... , numOutputs}
// GH 18.11.2023


#ifndef ANYNNET_H_
#define ANYNNET_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>


int numSets = 0;
float *x = NULL;			// input
float *y = NULL;			// output
float *test_x = NULL;		// for test data
float *test_y = NULL;		// for test data


typedef struct {
	int numNodes;
	int numInputs;
	float* a;
	float* z;
	float* b;
	float* w;
} Layer;

typedef struct {
	int numLayers;
	Layer* pLayer;
} Net;


float rand_float();
float sigmoidf(float x);
float d_sigmoidf_z(float z);
float reluf(float x);
float d_reluf(float a);

Net net_create(int* arch, int numElements);
void net_fill(Net net, float x);
void net_rand(Net net, float lo, float hi);
void net_print(Net net, const char* name);
void net_forward(Net net, float* ti, int set);
float net_cost(Net net, float* ti, float* to, int num_train);
void net_gradients(Net net, Net grd, float* ti, float* to, int numSet, int skip);
float net_verify(Net net, float* ti, float* to, int num_train, float limit, int print_level);
void net_learn(Net net, Net grd, float rate);
void net_read_mnist_data(const char* data_filename, const char* label_filename, bool bTrain);
void net_print_dataset(int set, int numInputs, int numOutputs);

#endif // ANYNNET_H_

#ifdef ANYNNET_IMPLEMENTATION

Net net_create(int* arch, int numElements)
{
	int num_layers = numElements - 1;
	Layer* pLayer = (Layer*)malloc(num_layers * sizeof(Layer));

	for (int l = 0; l < num_layers; l++) {
		int numNodes = arch[l + 1];
		int numInputs = arch[l];
		pLayer[l].numNodes = numNodes;
		pLayer[l].numInputs = numInputs;
		pLayer[l].w = (float*)malloc(numNodes * numInputs * sizeof(float));
		assert(pLayer[l].w != NULL);
		pLayer[l].b = (float*)malloc(numNodes * sizeof(float));
		assert(pLayer[l].b != NULL);
		pLayer[l].a = (float*)malloc(numNodes * sizeof(float));
		assert(pLayer[l].a != NULL);
		pLayer[l].z = (float*)malloc(numNodes * sizeof(float));
		assert(pLayer[l].z != NULL);
	}

	Net net;
	net.numLayers = num_layers;
	net.pLayer = pLayer;
	return net;
}

float rand_float()
{
	return (float)rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

float d_sigmoidf_z(float z)		// this is for z
{
	return (1.0f / (1.0f + expf(-z)) * (1.0f - (1.0f / (1.0f + expf(-z)))));
}

float reluf(float x)
{
	return x > 0 ? x : 0;
}

float d_reluf(float a)
{
	return a > 0 ? 1 : 0;
}

void net_fill(Net net, float x)
{
	for (int l = 0; l < net.numLayers; l++) {
		int numN = net.pLayer[l].numNodes;
		for (int n = 0; n < numN; n++) {
			int numI = net.pLayer[l].numInputs;
			for (int i = 0; i < numI; i++) {
				net.pLayer[l].w[n * numI + i] = x;
			}
			net.pLayer[l].b[n] = x;
			net.pLayer[l].a[n] = x;
			net.pLayer[l].z[n] = x;
		}
	}
}

void net_rand(Net net, float lo, float hi)
{
	for (int l = 0; l < net.numLayers; l++) {
		int numN = net.pLayer[l].numNodes;
		for (int n = 0; n < numN; n++) {
			int numI = net.pLayer[l].numInputs;
			for (int i = 0; i < numI; i++) {
				net.pLayer[l].w[n * numI + i] = rand_float() * (hi - lo) + lo;
			}
			net.pLayer[l].b[n] = rand_float() * (hi - lo) + lo;
		}
	}
}

void net_print(Net net, const char* name)
{
	printf("Net-Info: %s\n", name);
	int numL = net.numLayers;
	printf("numLayers: %d\n", numL);
	for (int l = 0; l < numL; l++) {
		printf("Layer: %d      ", l);
		int numN = net.pLayer[l].numNodes;
		printf(" numNodes: %d      ", numN);
		int numI = net.pLayer[l].numInputs;
		printf("numInputs: %d\n", numI);
		for (int n = 0; n < numN; n++) {
			printf("  Node: %d      ", n);
			for (int i = 0; i < numI; i++) {
				printf("w%d: %9.5f    ", i, net.pLayer[l].w[n * numI + i]);
			}
			printf("\n");
			printf("                b: %9.5f\n", net.pLayer[l].b[n]);
		}
	}
}

void net_forward(Net net, float* ti, int set)
{
	float z = 0;
	for (int l = 0; l < net.numLayers; l++) {
		for (int n = 0; n < net.pLayer[l].numNodes; n++) {
			z = 0;
			int numI = net.pLayer[l].numInputs;
			for (int i = 0; i < numI; i++) {
				float xx;
				if (l < 1) {
					xx = ti[set * numI + i];
				}
				else {
					xx = net.pLayer[l - 1].a[i];
				}
				z += xx * net.pLayer[l].w[n * numI + i];
			}
			z += net.pLayer[l].b[n];
			if (z > 1e10)
				printf("overflow in forward()\n");
			net.pLayer[l].z[n] = z;
			net.pLayer[l].a[n] = sigmoidf(z);
		}
	}
}

float net_cost(Net net, float* ti, float* to, int numSets)
{
	float diff = 0.0f;
	int numOut = net.pLayer[net.numLayers - 1].numNodes;
	for (int t = 0; t < numSets; t++) {
		net_forward(net, ti, t);
		for (int out = 0; out < numOut; out++) {
			float a = net.pLayer[net.numLayers - 1].a[out];
			float yy = to[t * numOut + out];
			diff += ((a - yy) * (a - yy));
			if (diff > 1e10)
				printf("overflow in cost()\n");
		}
	}
	return diff / numSets / numOut;
}

void net_print_dataset(int set, int numInputs, int numOutputs)
{
	for (int i = 0; i < numInputs; i++) {
		printf("%02.1f  ", x[set * numInputs + i]);
	}
	printf("\n");
	for (int out = 0; out < numOutputs; out++) {
		printf("%2.2f   ", y[set * numOutputs + out]);
	}
	printf("\n");
}

float net_verify(Net net, float* ti, float *to, int numSets, float limit, int print_level)
{
	int success = 0;
	int inc = 0;
	int numI = net.pLayer[0].numInputs;
	int numO = net.pLayer[net.numLayers - 1].numNodes;
	for (int t = 0; t < numSets; t++) {
		net_forward(net, ti, t);
		if (print_level == 2) {
			for (int i = 0; i < numI; i++) {
				printf("%01.0f  ", ti[t * numI + i]);
			}
			printf("    ");
		}
		if(print_level > 0){
			for (int out = 0; out < numO; out++) {
				printf("%2.2f ", net.pLayer[net.numLayers - 1].a[out]);
			}
			printf("   ");
			for (int out = 0; out < numO; out++) {
				printf("%1.0f ", to[t * numO + out]);
			}
			printf("\n");
		}
		inc = 0;
		for (int out = 0; out < numO; out++) {
			if (fabs(net.pLayer[net.numLayers - 1].a[out] - to[t * numO + out]) < limit) {
				inc++;
			}
		}
		if (inc == numO) {
			success++;
		}
	}
	return (float)success / (float)numSets;
}

void net_gradients(Net net, Net grd, float* ti, float* to, int numSets, int skip)
{
	float w, a, z, yy, da, da1, dz, dz1, dw, db;		// da, db, dw are dcost/d...
	int numL, numN, numI, numM;
	net_fill(grd, 0.0f);
	for (int t = 0; t < numSets; t++) {
		if(t == skip){						// option: skip 1 dataset and use as test
			continue;
		}
		net_forward(net, ti, t);
		numL = net.numLayers;
		for (int l = numL - 1; l >= 0; l--) {
			numN = net.pLayer[l].numNodes;
			for (int n = 0; n < numN; n++) {
				if (l == net.numLayers - 1) {
					a = net.pLayer[l].a[n];
					yy = to[t * numN + n];
					da = 2.0f * (a - yy);
				}
				else {
					da = 0;
					numM = net.pLayer[l + 1].numNodes;
					numI = net.pLayer[l + 1].numInputs;
					for (int m = 0; m < numM; m++) {
						da1 = grd.pLayer[l + 1].a[m];
						dz1 = grd.pLayer[l + 1].z[m];
						w = net.pLayer[l + 1].w[m * numI + n];
						da += da1 * dz1 * w;
					}
				}
				grd.pLayer[l].a[n] = da;							// dc/da
				z = net.pLayer[l].z[n];
				dz = d_sigmoidf_z(z);
				grd.pLayer[l].z[n] = dz;
				db = da * dz;
				grd.pLayer[l].b[n] += db;							// dc/db
				if (grd.pLayer[l].b[n] > 1e10)
					printf("overflow in gradients() b\n");

				numI = net.pLayer[l].numInputs;
				for (int i = 0; i < numI; i++) {
					if (l > 0) {
						a = net.pLayer[l - 1].a[i];
					}
					else {
						a = ti[t * numI + i];
					}
					dw = da * dz * a;
					grd.pLayer[l].w[n * numI + i] += dw;				// dc/dw
					if (grd.pLayer[l].w[n * numI + i] > 1e20)
						printf("overflow in gradients() w\n");
				}
			}
		}
	}
	// division by numSets
	for (int l = 0; l < net.numLayers; l++) {
		int numN = net.pLayer[l].numNodes;
		for (int n = 0; n < numN; n++) {
			int numI = net.pLayer[l].numInputs;
			for (int i = 0; i < numI; i++) {
				grd.pLayer[l].w[n * numI + i] /= numSets;
			}
			grd.pLayer[l].b[n] /= numSets;
		}
	}
}

void net_learn(Net net, Net grd, float rate)
{
	// subtract "rated" gradients from parameters
	for (int l = 0; l < net.numLayers; l++) {
		int numN = net.pLayer[l].numNodes;
		for (int n = 0; n < numN; n++) {
			int numI = net.pLayer[l].numInputs;
			for (int i = 0; i < numI; i++) {
				net.pLayer[l].w[n * numI + i] -= rate * grd.pLayer[l].w[n * numI + i];
			}
			net.pLayer[l].b[n] -= grd.pLayer[l].b[n];
		}
	}
}

int swapBytes(int i)
{
	unsigned char c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;
	return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void net_read_mnist_data(const char* data_filename, const char* label_filename, bool bTrain)
{
	// for idx3-ubyte, idx1-ubyte files
	FILE* file;
	file = fopen(data_filename, "rb");
	int magic_number;
	int numImages;
	int numRows;
	int numCols;
	unsigned char temp = 0;
	unsigned char *items = NULL;
	int cnt = 0;
	if (file)
	{
		printf("file: %s\n", data_filename);
		fread((char*)&magic_number, 1, sizeof(magic_number), file);
		magic_number = swapBytes(magic_number);
		printf("magic number: %d\n", magic_number);
		fread((char*)&numImages, 1, sizeof(numImages), file);
		numImages = swapBytes(numImages);
		numSets = numImages;
		printf("number of images: %d\n", numImages);
		fread((char*)&numRows, 1, sizeof(numRows), file);
		numRows = swapBytes(numRows);
		printf("number of rows: %d\n", numRows);
		fread((char*)&numCols, 1, sizeof(numCols), file);
		numCols = swapBytes(numCols);
		printf("number of cols: %d\n", numCols);
		if (bTrain) {
			x = (float*)malloc(numRows * numCols * numImages * sizeof(float));
			assert(x != NULL);
		}
		else {
			test_x = (float*)malloc(numRows * numCols * numImages * sizeof(float));
			assert(x != NULL);
		}
		for (int i = 0; i < numSets; i++) {
			for (int r = 0; r < numRows; r++) {
				for (int c = 0; c < numCols; c++) {
					fread((char*)&temp, 1, sizeof(temp), file);
					if (bTrain) {
						x[cnt++] = (float)temp / 255.0f;
					}
					else {
						test_x[cnt++] = (float)temp / 255.0f;
					}
				}
			}
		}
		fclose(file);
	}

	file = fopen(label_filename, "rb");
	if (file)
	{
		printf("file: %s\n", label_filename);
		fread((char*)&magic_number, 1, sizeof(magic_number), file);
		magic_number = swapBytes(magic_number);
		printf("magic number: %d\n", magic_number);
		fread((char*)&numImages, 1, sizeof(numImages), file);
		numImages = swapBytes(numImages);
		numSets = numImages;
		printf("number of labels: %d\n", numImages);
		if (bTrain) {
			y = (float*)malloc(numSets * 10 * sizeof(float));
			assert(y != NULL);
			memset(y, 0, numSets * 10 * sizeof(float));
		}
		else {
			test_y = (float*)malloc(numSets * 10 * sizeof(float));
			assert(test_y != NULL);
			memset(test_y, 0, numSets * 10 * sizeof(float));
		}
		for (int i = 0; i < numSets; i++) {
			fread((char*)&temp, 1, sizeof(temp), file);
			if (bTrain) {
				y[i * 10 + temp] = 1.0f;
			}
			else {
				test_y[i * 10 + temp] = 1.0f;
			}
		}
	}
}

#endif // ANYNNET_IMPLEMENTATION
