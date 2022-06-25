#include "Perceptron.h"

Perceptron::Perceptron(std::vector<int> npl){
    this->d.resize(npl.size());
    this->W = std::vector<Eigen::MatrixXd>(this->d.size());
    this->X = std::vector<Eigen::VectorXd>(this->d.size());
    this->deltas = std::vector<Eigen::VectorXd>(this->d.size());
    this->l = this->d.size() - 1;
    std::copy(npl.begin(), npl.end(), this->d.begin());

    this->W[0] = Eigen::MatrixXd::Zero(1, 1);

    for (int i = 0; i < this->d.size(); i++){
        if(i != 0){
            this->W[i] = Eigen::MatrixXd::Random(this->d[i - 1] + 1, this->d[i] + 1);
            this->W[i].col(0).setLinSpaced(0, 0);
        }
        this->X[i] = Eigen::VectorXd(this->d[i] + 1).setZero();
        this->X[i](0) = 1;

        this->deltas[i] = Eigen::VectorXd(this->d[i] + 1).setZero();
    }

    this->size_predict_result = this->X[this->l].size() - 1;
}

Eigen::VectorXd Perceptron::Predict(Eigen::VectorXd sample_inputs, bool is_classification) {
    for(int i = 0; i < this->d[0]; i++){
        this->X[0](i + 1) = sample_inputs(i);
    }

    double total = 0;
    for(int i = 1; i < l + 1; i++){
        total = 0;

        for(int j = 1; j < this->d[i] + 1; j++){
            for(int k = 0; k < this->d[i - 1] + 1; k++)
                total += this->W[i](k, j) * this->X[i - 1](k);

            if(i < this->l || is_classification)
                total = tanh(total);

            this->X[i](j) = total;
        }
    }

    Eigen::VectorXd result (this->size_predict_result);
    result << this->X[this->l].block(1, 0, this->X[this->l].size() - 1, 1);

    return result;
}

void Perceptron::Train(Eigen::MatrixXd all_sample_inputs,
                Eigen::MatrixXd all_samples_expected_outputs,
                float learning_rate,
                bool is_classification,
                int nb_iter) {

    std::srand(std::time(nullptr));
    double semi_gradient;
    Eigen::Index nb_rows;
    Eigen::VectorXd predict_result(this->X[this->l].size() - 1);

    for(int i = 0; i < nb_iter; i++){
        nb_rows = all_sample_inputs.rows();
        int k = rand() % nb_rows;
        Eigen::VectorXd sample_inputs(all_sample_inputs.row(k).size());
        Eigen::VectorXd sample_expected_output(all_samples_expected_outputs.row(k).size());

        sample_inputs = all_sample_inputs.row(k);
        sample_expected_output = all_samples_expected_outputs.row(k);

        predict_result << Predict(sample_inputs, is_classification);

        sample_inputs.resize(0);

        for(int j = 1; j < this->d[this->l] + 1; j++){
            semi_gradient = this->X[this->l](j) - sample_expected_output(j - 1);

            if(is_classification)
                semi_gradient *= (1 - pow(this->X[this->l](j), 2));

            this->deltas[this->l](j) = semi_gradient;
        }

        sample_expected_output.resize(0);

        for(long m = this->l; m > 0; m--){
            double total;
            for(int n = 1; n < this->d[m - 1] + 1; n++){
                total = 0.0;

                for(int o = 1; o < this->d[m] + 1; o++)
                    total += this->W[m](n, o) * this->deltas[m](o);
                total = (1 - pow(this->X[m - 1](n), 2)) * total;
                this->deltas[m - 1](n) = total;
            }
        }

        for(int m = 1; m < this->l + 1; m++){
            for(int n = 0; n < this->d[m - 1] + 1; n++){
                for (int o = 0; o < this->d[m] + 1; o++) {
                    this->W[m](n, o) -= learning_rate * this->X[m - 1](n) * this->deltas[m](o);
                }
            }
        }
    }

    predict_result.resize(0);
}

void Perceptron::printWeights() {
    for(int i = 0; i < this->W.size(); i++){
        std::cout << this->X[i] << std::endl;
        std::cout << "\n" << std::endl;
    }
}

void Perceptron::Destroy() {
    this->d.resize(0);

    for(int i = 0; i < this->W.size(); i++){
        this->W[i].resize(0,0);
    }
    this->W = std::vector<Eigen::MatrixXd>();
    for(int i = 0; i < this->X.size(); i++){
        this->X[i].resize(0);
    }
    this->X = std::vector<Eigen::VectorXd>();
    for(int i = 0; i < this->deltas.size(); i++){
        this->deltas[i].resize(0);
    }
    this->deltas = std::vector<Eigen::VectorXd>();
}

