#ifndef DLLEXEC_CLUSTER_H
#define DLLEXEC_CLUSTER_H

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "Point.h"

class Cluster {
public:
    explicit Cluster(int clusterId, Point centroid);
    void addPoint(Point p);
    bool removePoint(int pointId);
    void removeAllPoints();
    int getId();
    Point getPoint(int pos);
    int getSize();
    double getCentroidByPos(int pos);
    void setCentroidByPos(int pos, double val);


private:
    int clusterId;
    std::vector<double> centroid;
    std::vector<Point> points;
};


#endif //DLLEXEC_CLUSTER_H
