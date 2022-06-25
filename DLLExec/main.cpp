#include <iostream>
#include <Eigen/Dense>
#include "Perceptron.h"

int main() {

    /* ------------------------------------------------------------------ */

    Eigen::MatrixXd X {
            {0, 0},
            {1, 0},
            {0, 1},
            {1, 1}
    };

    Eigen::MatrixXd Y {
            {-1},
            {1},
            {1},
            {-1}
    };

    Eigen::MatrixXd X2 {
            {0, 0},
            {1, 0},
            {0, 1},
            {1, 1}
    };

    Eigen::MatrixXd Y2 {
            {42},
            {8},
            {3},
            {2}
    };

    std::vector<int> npl{2, 3, 1};

    /* ------------------------------------------------------------------ */

    Perceptron mpl = Perceptron(npl);

    std::cout << "\n" << "For classification : " << "\n" << std::endl;

    int nb_iter = 10000;
    std::cout << "Before : " << std::endl;
    for(int i = 0; i < X.rows(); i++){
        std::cout << mpl.Predict((Eigen::VectorXd)X.row(i), true) << std::endl;
    }

    mpl.Train(X, Y, 0.01, true, nb_iter);
    std::cout << "After : " << std::endl;
    for(int i = 0; i < X.rows(); i++){
        std::cout << mpl.Predict((Eigen::VectorXd)X.row(i), true) << std::endl;
    }

    std::cout << "\n" << "For regression : " << "\n" << std::endl;

    std::cout << "Before : " << std::endl;
    for(int i = 0; i < X2.rows(); i++){
        std::cout << mpl.Predict((Eigen::VectorXd)X2.row(i), false) << std::endl;
    }

    mpl.Train(X2, Y2, 0.01, false, nb_iter);
    std::cout << "After : " << std::endl;
    for(int i = 0; i < X2.rows(); i++){
        std::cout << mpl.Predict((Eigen::VectorXd)X2.row(i), false) << std::endl;
    }

    mpl.Destroy();

    return 0;
}
