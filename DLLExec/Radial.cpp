#include "Radial.h"


Eigen::MatrixXd kMeansClustering(Eigen::MatrixXd inputs, int epochs, int k){
    srand(time(0));

    Eigen::MatrixXd centroids(k, inputs.cols());

    for(int i = 0; i < k; i++){
        centroids.row(i) = inputs.row(rand() % inputs.rows());
    }

    Eigen::MatrixXd points_clusters(k, Eigen::NoChange);

    //std::cout << cluster_groups << std::endl;
    double distance;
    double max_distance;
    int centroid_index;

    for(int point = 0; point < inputs.rows(); point++){
        max_distance = 0;
        for(int c = 0; c < centroids.rows(); c++){
            distance = euclideanDistance(centroids.row(c), inputs.row(point));
            if(distance > max_distance) {
                max_distance = distance;
                centroid_index = c;
            }
        }
        printf("La distance max de la ligne %d est %f", point, max_distance);
        points_clusters(point) = centroid_index;
    }
    std::cout << points_clusters << std::endl;

    return centroids;
}

Eigen::MatrixXd kMeansClustering2(Eigen::MatrixXd inputs, int epochs, int k){
    srand(time(0));

    std::vector<Point> points;
    for(int in = 0; in < inputs.rows(); in++){
        points.emplace_back(inputs.row(in));
    }
    std::cout << points[0].coords << std::endl;
    for (std::vector<Point>::iterator it = points.begin() ; it != points.end(); ++it)
        std::cout << ' ' << it->coords;

    Eigen::MatrixXd centroids(k, inputs.cols());

    for(int i = 0; i < k; i++){
        centroids.row(i) = inputs.row(rand() % inputs.rows());
    }
    return centroids;
}

double euclideanDistance(Eigen::VectorXd p, Eigen::VectorXd q){
    double total_sum = 0;

    for(int i = 0; i < p.rows(); i++){
        total_sum += pow(q(i) - p(i), 2);
    }

    return sqrt(total_sum);
}