#ifndef DLLEXEC_RADIAL_H
#define DLLEXEC_RADIAL_H

#include <ctime>
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "Point.h"

Eigen::MatrixXd kMeansClustering(Eigen::MatrixXd inputs, int epochs, int k);
Eigen::MatrixXd kMeansClustering2(Eigen::MatrixXd inputs, int epochs, int k);
double euclideanDistance(Eigen::VectorXd p, Eigen::VectorXd q);


#endif //DLLEXEC_RADIAL_H
