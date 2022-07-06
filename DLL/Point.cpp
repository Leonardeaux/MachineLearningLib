#include "Point.h"

Point::Point(int id, Eigen::VectorXd dataCoords) {
    pointId = id;
    values.resize(dataCoords.size());
    Eigen::VectorXd::Map(&values[0], dataCoords.size()) = dataCoords;
    dimensions = (int)values.size();
    clusterId = -1;
}

int Point::getDimensions() {
    return dimensions;
}

int Point::getCluster() {
    return clusterId;
}

int Point::getID() {
    return pointId;
}

void Point::setCluster(int val) {
    clusterId = val;
}

double Point::getVal(int pos) {
    return values[pos];
}


