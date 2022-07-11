#include <iostream>
#include <Eigen/Dense>
#include "Perceptron.h"
#include "KMeans.h"
#include "Point.h"

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

    Eigen::MatrixXd X3 {
            {2, 3, 4, 5, 4, 5},
            {1, 3, 2, 2, 4, 9},
            {0, 1, 1, 4, 0, 0},
            {1, 1, 1, 1, 1, 1}
    };

    Eigen::VectorXd P1 {
            {1},
            {4},
            {1},
            {4}
    };

    Eigen::VectorXd P2 {
            {2},
            {2},
            {2},
            {2}
    };

    Eigen::MatrixXd K {
            {9, 9, 9},
            {1, 1, 5},
            {-1, -1, 6},
            {3, 3, 4},
            {10, 10, 9},
            {-2, -2, 0},
            {7, 8, 8},
            {0.2, 0, 5},
            {-1, 0, 8},
            {6, 10, 11}
    };

    Eigen::MatrixXd X4 {
            {1.72,  1.73},
            {-8.08,  7.19},
            {-8.83,  6.50},
            {-8.34,  8.88},
            {-8.93, -4.68},
            { 5.32, -8.31},
            {-8.92, -8.81},
            { 8.00,  6.60},
            {-2.25,  6.67},
            {-6.31, -1.20},
            {-5.99,  9.45},
            {-1.66,  4.42},
            { 7.28, -1.90},
            { 9.01, -7.91},
            {-5.28, -1.01},
            {-7.41,  9.45},
            {-4.63, -3.58},
            {-9.04, -3.57},
            {-1.39,  7.98},
            {-9.69, -2.13},
            { 6.82, -9.99},
            { 9.47,  6.53},
            {-2.97, -2.34},
            {-6.37, -8.63},
            { 8.06, -1.62},
            { 4.25,  5.01}
    };

    Eigen::MatrixXd Y4 {
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 0},
            {0, 0, 1},
            {0, 1, 0},
            {0, 1, 0},
            {0, 0, 1},
            {0, 1, 0}
    };

    double array[20] = {9, 9, 1, 1, -1, -1, 3, 3, 10, 10, -2, -2, 7, 8, 0.2, 0, -1, 0, 6, 10};

    std::vector<int> npl{2, 3};

    /* ------------------------------------------------------------------ */

    Perceptron mpl = Perceptron(npl);

    std::cout << "\n" << "For classification : " << "\n" << std::endl;

    int nb_iter = 1000000;
    std::cout << "Before : " << std::endl;
    for(int i = 0; i < X4.rows(); i++){
        std::cout << mpl.Predict((Eigen::VectorXd)X4.row(i), true) << std::endl;
    }

    mpl.Train(X4, Y4, 0.01, true, nb_iter);
    std::cout << "After : " << std::endl;
    for(int i = 0; i < X4.rows(); i++){
        std::cout << mpl.Predict((Eigen::VectorXd)X4.row(i), true) << std::endl;
    }

/*    std::cout << "\n" << "For regression : " << "\n" << std::endl;

    std::cout << "Before : " << std::endl;
    for(int i = 0; i < X2.rows(); i++){
        std::cout << mpl.Predict((Eigen::VectorXd)X2.row(i), false) << std::endl;
    }

    mpl.Train(X2, Y2, 0.01, false, nb_iter);
    std::cout << "After : " << std::endl;
    for(int i = 0; i < X2.rows(); i++){
        std::cout << mpl.Predict((Eigen::VectorXd)X2.row(i), false) << std::endl;
    }

    mpl.Destroy();*/


    //std::cout << "\n" << predictLinearModelRegression(X, Y) << "\n" << std::endl;

    //std::cout << "\n" << kMeansClustering(X3, 100, 3) << "\n" << std::endl;

//    std::vector<Point> all_points;
//    for(int i = 0; i < K.rows(); i++){
//        Point point(i + 1, K.row(i));
//        all_points.push_back(point);
//    }
//
//    std::cout << all_points[9].getVal(1) << std::endl;
//
//    int iters = 100;
//
//    KMeans kMeans(2, iters);
//    kMeans.run(all_points);
//
//    std::cout << "Clusters : " << std::endl;
//
//    std::vector<Cluster> all_clusters = kMeans.getClusters();
//
//    for (auto it = all_clusters.begin() ; it != all_clusters.end(); ++it){
//        for(int i = 0; i < 2; i++){
//            std::cout << ' ' << it->getCentroidByPos(i);
//        }
//        std::cout << '\n';
//    }

//   auto *res = new double[100];
//
//   res = kmeans_centroids(array, 10, 2, 2, 100);

//    kMeansClustering2(X2, 190, 3);
//    std::cout << "\n" << kMeansClustering(X2, 190, 3) << "\n" << std::endl;

//    Eigen::MatrixXd test(1, Eigen::NoChange);
//
//    std::cout << "\n" << test.rows() << "\n" << std::endl;
//    for(int i = 0; i < 5; i++){
//        test.conservativeResize(Eigen::NoChange, test.cols()+1);
//        test(0, i) = 1.0;
//    }
//    std::cout << "\n" << test.row(0) << "\n" << std::endl;

    return 0;
}