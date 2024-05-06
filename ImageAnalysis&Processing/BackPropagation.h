#pragma once

#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <vector>

#define LAMBDA 1.0
#define ETA 0.1

#define SQR( x ) ( ( x ) * ( x ) )

struct NN {
	int* n; // pocty neuronu
	int l; // pocet vrstev
	double*** w; // vahy

	double* in; // vstupni vektor
	double* out; // vystupni vektor
	double** y; // vystupni vektory vrstev

	double** d; // chyby neuronu
};

void randomize(double* w, int n);
NN* createNN(int n, int h, int o);
void releaseNN(NN*& nn);
void feedForward(NN* nn);
double backPropagation(NN* nn, double* t);
void setInput(NN* nn, double* in, bool verbose = false);
int getOutput(NN* nn, bool verbose = false);
void train(NN* nn);
void test(NN* nn, int num_samples);

#endif // BACKPROPAGATION_H