#ifndef DLLEXEC_POINT_H
#define DLLEXEC_POINT_H

#include <ctime>
#include <vector>
#include <Eigen/Dense>
#include <iostream>

class Point {
public:
    explicit Point(int id, Eigen::VectorXd dataCoords);
    int getDimensions();
    int getCluster();
    int getID();
    void setCluster(int val);
    double getVal(int pos);

private:
    int pointId, clusterId;
    int dimensions;
    std::vector<double> values;
};


#endif //DLLEXEC_POINT_H
