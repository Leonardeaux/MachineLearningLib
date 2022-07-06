#include "KMeans.h"

KMeans::KMeans(int K, int iterations) {
    this->K = K;
    this->iters = iterations;
}

void KMeans::run(std::vector<Point> &all_points) {
    total_points = (int)all_points.size();
    dimensions = all_points[0].getDimensions();

    std::vector<int> used_pointIds;

    for(int i = 1; i <= K; i++){
        while(true){
            int index = rand() % total_points;

            if(find(used_pointIds.begin(), used_pointIds.end(), index) == used_pointIds.end()){
                used_pointIds.push_back(index);
                all_points[index].setCluster(i);
                Cluster cluster(i, all_points[index]);
                clusters.push_back(cluster);
                break;
            }
        }
    }
    std::cout << "Clusters initialized = " << clusters.size() << std::endl << std::endl;

    std::cout << "Running K-Means Clustering.." << std::endl;

    int iter = 1;

    while(true){
        std::cout << "Iter - " << iter << "/" << iters << std::endl;
        bool done = true;

        // Add all points to their nearest cluster
        #pragma omp parallel for reduction(&&: done) num_threads(16)
        for(int i = 0; i < total_points; i++){
            int currentClusterId = all_points[i].getCluster();
            int nearestClusterId = getNearestClusterId(all_points[i]);

            if(currentClusterId != nearestClusterId){
                all_points[i].setCluster(nearestClusterId);
                done = false;
            }
        }

        // Clear all existing clusters
        clearClusters();

        // Reassign points to their new clusters
        for(int i = 0; i < total_points; i++){
            // Cluster index is ID-1
            clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
        }

        // Recalculating the center of each cluster
        for(int i = 0; i < K; i++){
            int clusterSize = clusters[i].getSize();

            for(int j  = 0; j < dimensions; j++){
                double sum = 0.0;
                if(clusterSize > 0){
                    #pragma omp parallel for reduction(+: sum) num_threads(16)
                    for(int p = 0; p < clusterSize; p++){
                        sum += clusters[i].getPoint(p).getVal(j);
                    }
                    clusters[i].setCentroidByPos(j, sum / clusterSize);
                }
            }
        }

        if(done || iter >= iters){
            std::cout << "Clustering completed in iteration : " << iter << std::endl;
            break;
        }
        iter++;
    }

    for(int i = 0; i < K; i++){
        std::cout << "Cluster " << clusters[i].getId() << " centroid : ";
        for (int j = 0; j < dimensions; j++)
        {
            std::cout << clusters[i].getCentroidByPos(j) << " ";    // Output to console
        }
        std::cout << std::endl;
    }
}

void KMeans::clearClusters() {
    for(int i = 0; i < K; i++){
        clusters[i].removeAllPoints();
    }
}

int KMeans::getNearestClusterId(Point point) {
    double sum = 0.0;
    double min_dist;
    int NearestClusterId;

    if(dimensions == 1){
        min_dist = abs(clusters[0].getCentroidByPos(0) - point.getVal(0));
    } else {
        for(int i = 0; i < dimensions; i++){
            sum += pow(clusters[0].getCentroidByPos(i) - point.getVal(i), 2.0);
        }
        min_dist = sqrt(sum);
    }
    NearestClusterId = clusters[0].getId();

    for(int i = 1; i < K; i++){
        double dist;
        sum = 0.0;

        if(dimensions == 1){
            dist = abs(clusters[i].getCentroidByPos(0) - point.getVal(0));
        } else {
            for(int j =  0; j < dimensions; j++){
                sum += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
            }

            dist = sqrt(sum);
        }
        if(dist < min_dist){
            min_dist = dist;
            NearestClusterId = clusters[i].getId();
        }
    }

    return NearestClusterId;
}

std::vector<Cluster> KMeans::getClusters() {
    return clusters;
}
