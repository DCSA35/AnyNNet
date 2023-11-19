// main.c
// >>Machine Learning in C without Matrices<<
// based on Tsoding Daily (https://www.youtube.com/watch?v=PGSba51aRYU&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw&index=1)
// process any fully connected system: arch{numInputs, numNodes of layer i, .... , numOutputs}
// linux: gcc main.c -o main -lm
// windows: VS (ISO C11)
// GH 13.11.2023

#define _CRT_SECURE_NO_WARNINGS

#define ANYNNET_IMPLEMENTATION
#include "AnyNNet.h"


// select example you want to try and select data loading function calls in main() respectively

// XOR
float xk[] = {0, 0, 0, 1, 1, 0, 1, 1};
float yk[] = {0, 1, 1, 0};
// uncomment the following line and select the data loader in main()
int arch[] = {2, 2, 1};

// Stairs (https://github.com/ben519/MLPB/tree/master/Problems/Classify%20Images%20of%20Stairs)
// rate: 1
// random: -1, +1
// epochs = 4000 (success > 95%)
// uncomment the following and the data loader call in main()
//int arch[] = {4, 4, 1};

// Binary Adder (https://www.youtube.com/watch?v=o7da9anmnMs&list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw&index=3)
#define BITS 4
// rate: 1
// random: -1, +1
// epochs: 4000  (success 100%)
// uncomment the following line and select the data creator call in main()
//int arch[] = {2*BITS, 4*BITS, BITS+1};

// MNIST (http://yann.lecun.com/exdb/mnist/)
// rate: 1.0f
// random: -0.1f, +0.1f
// epochs: > 1500
// uncomment the following line and select the data creator call in main()
//int arch[] = {784, 10};

// Zalando (https://github.com/zalandoresearch/fashion-mnist)
// uncomment the following line and select the data creator call in main()
//int arch[] = { 784, 10 };


void load_xor_data()
{
	x = xk;
	y = yk;
	numSets = 4;
}

void create_adder_data()
{
	numSets = 1 << (2 * BITS);

	x = (float*)malloc(numSets * 2 * BITS * sizeof(float));
	assert(x != NULL);
	y = (float*)malloc(numSets * (BITS + 1) * sizeof(float));
	assert(y != NULL);
	int t;
	for (int i = 0; i < numSets; i++) {
		t = i;
		for (int j = 0; j < 2 * BITS; j++) {
			x[i * 2 * BITS + j] = (float)(t % 2);
			//printf("%d  ", t);
			t >>= 1;
		}
		//printf("\n");
		int hi = (i & (numSets - BITS * BITS)) >> BITS;
		int lo = i & ((1 << BITS) - 1);
		int a = lo + hi;
		//printf("%d   %d\n", lo, hi);
		int b = a;
		for (int j = 0; j < BITS + 1; j++) {
			if (j == BITS) {
				if (b >= (1 << BITS))
					y[i * (BITS + 1) + j] = 1;
				else
					y[i * (BITS + 1) + j] = 0;
				continue;
			}
			y[i * (BITS + 1) + j] = (float)(a % 2);
			a >>= 1;
		}
	}
	for (int i = 0; i < numSets; i++) {
		for (int j = 0; j < 2 * BITS; j++) {
			printf("%1.0f   ", x[i * 2 * BITS + j]);
		}
		for (int j = 0; j < BITS + 1; j++) {
			printf(" %1.0f  ", y[i * (BITS + 1) + j]);
		}
		printf("\n");
	}
}

int net_read_stairs_data(const char* filename, int numInputs, int numOutputs, bool bTrain)
{
	// format: (first line comment)
	// # of set, input-data (numInput), output-data (numOutputs)
	// returns numSets
	// mallocates memory for the data
	FILE* file;
	file = fopen(filename, "r");
	if (file == NULL) {
		printf("Cannot open File\n");
		return -1;
	}
	// count numSets
	int cnt = 0;
	char line[128];
	while (fscanf(file, "%s", line) != EOF) {
		cnt++;
	}
	rewind(file);
	numSets = cnt - 1;
	int inp = numInputs * numSets;
	if (bTrain) {
		x = (float*)malloc(inp * sizeof(float));
		assert(x != NULL);
	}
	else {
		test_x = (float*)malloc(inp * sizeof(float));
		assert(test_x != NULL);
	}
	int out = numOutputs * numSets;
	if (bTrain) {
		y = (float*)malloc(out * sizeof(float));
		assert(y != NULL);
	}
	else {
		test_y = (float*)malloc(inp * sizeof(float));
		assert(test_y != NULL);
	}

	int numToRead = numInputs + numOutputs + 1;		// line number

	int cntx = 0;
	int cnty = 0;
	int num = 0;
	char seps[4];
	strcpy(seps, ",");
	char* tok;
	float* val = (float*)malloc((numInputs + numOutputs) * sizeof(float));
	while (fscanf(file, "%s", line) != EOF) {
		if (line[0] > 57) {
			continue;
		}
		tok = strtok(line, seps);					// skip line number
		for (int i = 0; i < numToRead - 1; i++) {
			tok = strtok(NULL, seps);
			if (tok != NULL) {
				val[i] = (float)atof(tok);
			}
		}
		for (int i = 0; i < numToRead - 1; i++) {
			if (i < numInputs) {
				if (bTrain) {
					x[cntx++] = val[i] / 255.0f;
				}
				else {
					test_x[cntx++] = val[i] / 255.0f;
				}
			}
			else {
				if (bTrain) {
					y[cnty++] = val[i];
				}
				else {
					test_y[cnty++] = val[i];
				}
			}
		}
	}
	fclose(file);
	return cnt - 1;
}

int main()
{
	// select data loader here:
	load_xor_data();
	//numSets = net_read_stairs_data("train.csv", 4, 1, true);
	//create_adder_data();
	//net_read_mnist_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", true);	// MNIST
	//net_read_mnist_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", true);	// Zalando
	if (numSets < 1) {
		printf("No data set\n");
		exit(0);
	}
	bool bTestData = false;
	//bool bTestData = true;

	float rate = 1.0f;
	//float rate = 0.1f;
	int arch_count = sizeof(arch) / sizeof(arch[0]);
	Net net = net_create(arch, arch_count);
	Net grd = net_create(arch, arch_count);		// gradients
	srand((int)time(0));
	//net_rand(net, -0.1f, 0.1f);
	net_rand(net, -1, 1);
	//numSets = 500;			// optional set restriction 
	int epochs = 4000;
	int skip = numSets;			// no skip
	//skip = 120;				// optional training set to skip and use as test
	printf("cost: %1.5f\n", net_cost(net, x, y, numSets));
	clock_t sta = clock() * 1000 / CLOCKS_PER_SEC;
	for (int i = 0; i < epochs; i++) {
		net_gradients(net, grd, x, y, numSets, skip);
		net_learn(net, grd, rate);
	}
	clock_t end = clock() * 1000 / CLOCKS_PER_SEC;
	// select print level:
	int print_level = 2;
	//int print_level = 1;
	//int print_level = 0;
	float suc = net_verify(net, x, y, numSets, 0.5f, print_level);
	printf("Use train-data:\n");
	printf("numSets: %d\n", numSets);
	printf("cost: %1.5f\n", net_cost(net, x, y, numSets));
	printf("success: %1.2f%%\n", suc * 100);
	printf("finished %d epochs in %zu ms\n", epochs, end - sta);

	if (bTestData) {
		// use test data
		printf("Use test-data:\n");
		//numSets = net_read_stairs_data("test.csv", 4, 1, false);
		net_read_mnist_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", false);	// MNIST
		printf("numSets: %d\n", numSets);
		suc = net_verify(net, test_x, test_y, numSets, 0.5f, print_level);
		printf("test cost: %1.5f\n", net_cost(net, x, y, numSets));
		printf("test success: %1.2f%%\n", suc * 100);
		printf("finished %d epochs in %zu ms\n", epochs, end - sta);
	}

	//	if(suc > 0.99f) {
	//		net_print(net, "net");
	//	}

	if (skip < numSets) {
		printf("test skip: \n");
		net_forward(net, x, skip);

		int numI = net.pLayer[0].numInputs;
		for (int i = 0; i < numI; i++) {
			printf("%1.0f  ", x[skip * numI + i]);
		}
		printf("    ");
		int numOut = net.pLayer[net.numLayers - 1].numNodes;
		for (int out = 0; out < numOut; out++) {
			printf("%1.2f   ", net.pLayer[net.numLayers - 1].a[out]);
		}
		printf("   ");
		int inc = 0;
		for (int out = 0; out < numOut; out++) {
			printf("%1.0f   ", y[skip * numOut + out]);
		}
	}
	//getchar();		// for Standalone Windows Console
	return 0;
}
