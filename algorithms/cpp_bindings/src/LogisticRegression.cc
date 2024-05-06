#include <iostream>
#include<stdio.h>
#include<string.h>
#include<random>
#include<chrono>
#include<iostream>
#include<omp.h>

#include "cpp_bindings/include/LogisticRegression.h"

using namespace std::chrono;
using namespace std;

#define TOTAL_THREADS 8

// int N = 10, D = 3, x0 = 769224786, x1 = 988356581, A = 1272428648, B = 1988466161, C = 1956695325, M = 1000000007;
int N = 100000, D= 1600, x0 = 1736561047, x1 = 1688059698, A = 1346499669, B = 2123142289, C = 1835109006, M = 1000000007;

LogisticRegression::LogisticRegression() {}

template<class T>
T* generate_input(int x0,int x1, int A,int B,int C,int M,size_t size){
  T* ret = new T[size];
  ret[0] = x0 % M;
  ret[1] = x1 % M;
  for(size_t i = 2; i < size; ++i)
    ret[i] = (long long)((long long)A * ret[i - 1] + (long long)B * ret[i - 2] + C) % M;
  return ret;
}

double sigmoid(double X){
  return 1.0 / (1 + exp(-X));
}

// Creates the features of LR, a Vector
double* init_parameters(int D){
  double* ret = new double[D];
  std::default_random_engine generator;
  std::normal_distribution<double> distribution(0.0,1.0);
  for(int i = 0;i < D;++i){
    ret[i] = distribution(generator);
  }
  return ret;
}

// Forward selection for model
double forward_select(double* X,int D,double* P,double* grad,double Y){
  
  double dot_pdt = 0, pred;
  
  for(int i = 0;i < D;++i){
    dot_pdt += X[i] * P[i];
  }
  
  pred = sigmoid(dot_pdt);

  for(int i = 0; i < D; ++i){
    grad[i] += 1.0 / N * (pred - Y) * X[i];
  }

  return pred;
}

double learning_rate_schedule(int epoch){
  if(epoch < 100)
    return 0.15;
  if(epoch < 10000)
    return 0.1;
  else if(epoch < 20000)
    return 0.01;
  return 0.001;
}

// Generates labels
int* generate_label(double* X,int N,int D){
  int* ret = new int[N];

  for(int i = 0;i < N;++i){
    double tmp = 0;
    for(int j = 0;j < D;++j){
      if(j % 2)
        tmp += X[i * D + j] * X[i * D + (j + x0) % D];
      else
        tmp -= X[i * D + j] * X[i * D + (j + x1) % D];
    }
    ret[i] = round(sigmoid(tmp));
  }

  return ret;
}

double lr_compute(double* X, int N, int* Y, int D, double* grad, double* P, \
double* pred, float tolerance){

  
  int correct;

  double acc = 0.0;

  // Initial run
  for(int i = 0;i < N;++i)
      pred[i] = forward_select(X + i * D,D,P, grad, Y[i]);
    
  memset(grad, 0, D*sizeof(grad[0]));

  for(int epoch = 0;;++epoch){
    correct = 0;
    #pragma omp parallel for reduction(+:correct, grad[:D]) num_threads(omp_get_thread_num())
    for(int i = 0;i < N;++i){
      pred[i] = forward_select(X + i * D,D,P, grad, Y[i]);
      if(round(pred[i]) == Y[i])
        ++correct;
    }

    double learningRate = learning_rate_schedule(epoch);
    for(int i = 0;i < D;++i){
      P[i] -= learningRate * grad[i];
      grad[i] = 0;
    }

    acc = correct * 1.0 / N;
    if(acc > tolerance){
      printf("Done with Accuracy %f at epoch %d\n",acc,epoch);
      break;
    }
  }

  return acc;
}

double LogisticRegression::train()
{
  
  FILE* fout = fopen("params.out","w");
  
  auto start_ipgen = high_resolution_clock::now();
  double* X = generate_input<double>(x0,x1,A,B,C,M,(size_t)N * D);
  auto end_ipgen = high_resolution_clock::now();
  auto duration_ipgen = duration_cast<milliseconds>(end_ipgen - start_ipgen);
  std::cout << "LR Input Generation Duration:\t" << duration_ipgen.count() << " milliseconds" << endl;


  for(size_t i = 0;i < (size_t)N * D;++i)
    X[i] = X[i] * 1.0 / M;

  auto start_labelgen = high_resolution_clock::now();
  int* Y = generate_label(X,N,D);
  auto end_labelgen = high_resolution_clock::now();
  auto duration_labelgen = duration_cast<milliseconds>(end_labelgen - start_labelgen);
  std::cout << "LR Input Generation Duration:\t" << duration_labelgen.count() << " milliseconds" << endl;


  double* P = init_parameters(D);
  double tolerance = 0.6;

  double* pred = new double[N];
  double* grad = new double[D];

  auto start_lr = high_resolution_clock::now();
  double acc = lr_compute(X, N, Y, D, grad, P, pred, tolerance);
  auto end_lr = high_resolution_clock::now();
  auto duration_lrcal = duration_cast<milliseconds>(end_lr - start_lr);
  std::cout << "LR Calculation Duration:\t" << duration_lrcal.count() << " milliseconds" << endl;


  for(int i = 0;i < D;++i){
    fprintf(fout,"%.15f\n",P[i]);
  } 

  fclose(fout);

  delete[] X;
  delete[] Y;
  delete[] P;
  delete[] pred;
  
  return acc;
}


double train() {
  auto lr = new LogisticRegression();

  return lr->train();
}
