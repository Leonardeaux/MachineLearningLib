#ifndef DLLEXEC_KMEANS_H
#define DLLEXEC_KMEANS_H

#include <omp.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "Point.h"
#include "Cluster.h"

class KMeans {
public:
    explicit KMeans(int K, int iterations);
    void run(std::vector<Point> &all_points);
    std::vector<Cluster> getClusters();


private:
    int K, iters, dimensions, total_points;
    std::vector<Cluster> clusters;
    void clearClusters();
    int getNearestClusterId(Point point);

};


#endif //DLLEXEC_KMEANS_H
