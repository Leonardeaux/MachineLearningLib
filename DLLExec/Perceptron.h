#ifndef DLLEXEC_PERCEPTRON_H
#define DLLEXEC_PERCEPTRON_H

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <vector>

class Perceptron {
public:
    long size_predict_result;
    explicit Perceptron(std::vector<int> npl);
    Eigen::VectorXd Predict(Eigen::VectorXd sample_inputs, bool is_classification);
    void Train(Eigen::MatrixXd all_sample_inputs,
               Eigen::MatrixXd all_samples_expected_outputs,
               float learning_rate = 0.01,
               bool is_classification = true,
               int nb_iter = 10000);
    void printWeights();
    void Destroy();

private:
    int32_t l;
    std::vector<int> d;
    std::vector<Eigen::MatrixXd> W;
    std::vector<Eigen::VectorXd> X;
    std::vector<Eigen::VectorXd> deltas;
};


#endif //DLLEXEC_PERCEPTRON_H
