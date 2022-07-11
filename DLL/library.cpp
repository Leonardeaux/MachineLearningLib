#if WIN32
#define DLLEXPORT __declspec(dllexport)
#elif
#define DLLEXPORT
#endif

#include "library.h"

extern "C" {

    // Perceptron
DLLEXPORT Perceptron *create_mlp_model(int32_t *npl, int32_t npl_size) {
    std::cout << "Beginning create_mlp_model" << std::endl;

    std::vector<int> a_vector(npl, npl + npl_size);
    Perceptron *model = new Perceptron(a_vector);

    a_vector = std::vector<int>(); //Destroy vector
    return model;
}

DLLEXPORT int32_t get_predict_size(Perceptron *model) {
    return model->size_predict_result;
}

DLLEXPORT double *predict_mlp_model(Perceptron *model, double *inputs, size_t inputs_size, int32_t is_classification) {
    Eigen::VectorXd sample_inputs(inputs_size);
    for(size_t i = 0; i < inputs_size; i++){
        sample_inputs((int)i) = inputs[i];
    }

    Eigen::VectorXd result(model->size_predict_result);
    result = model->Predict(sample_inputs, is_classification);
    auto result_array = new double[model->size_predict_result];

    for(auto i = 0; i < model->size_predict_result; i++){
        result_array[i] = result(i);
    }

    sample_inputs.resize(0);
    result.resize(0);
    return result_array;
}

DLLEXPORT void train_mlp_model(Perceptron *model,
                               const double *all_inputs,
                               size_t inputs_size_rows,
                               size_t inputs_size_cols,
                               const double *all_outputs,
                               size_t outputs_size_rows,
                               size_t outputs_size_cols,
                               float learning_rate,
                               int32_t is_classification,
                               int32_t nb_iter) {
    Eigen::MatrixXd inputs(inputs_size_rows, inputs_size_cols);
    Eigen::MatrixXd outputs(outputs_size_rows, outputs_size_cols);
    int cpt = 0;
    for (size_t i = 0; i < inputs_size_rows; i++) {
        for (size_t j = 0; j < inputs_size_cols; j++) {
            inputs((int)i, (int)j) = all_inputs[cpt];
            cpt++;
        }
    }

    cpt = 0;
    for (size_t i = 0; i < outputs_size_rows; i++) {
        for (size_t j = 0; j < outputs_size_cols; j++) {
            outputs((int)i, (int)j) = all_outputs[cpt];
            cpt++;
        }
    }

    model->Train(inputs, outputs, learning_rate, is_classification, nb_iter);

    inputs.resize(0,0);
    outputs.resize(0,0);
}

    // RBF
DLLEXPORT double *kmeans_centroids(double *inputs, size_t inputs_rows, size_t inputs_cols, int32_t k, int32_t nb_iters) {
    Eigen::MatrixXd all_inputs(inputs_rows, inputs_cols);

    int cpt = 0;
    for (size_t i = 0; i < inputs_rows; i++) {
        for (size_t j = 0; j < inputs_cols; j++) {
            all_inputs((int)i, (int)j) = inputs[cpt];
            cpt++;
        }
    }

    std::vector<Point> all_points;
    for(int i = 0; i < all_inputs.rows(); i++){
        Point point(i + 1, all_inputs.row(i));
        all_points.push_back(point);
    }

    all_inputs.resize(0, 0);

    KMeans kMeans(k, nb_iters);
    kMeans.run(all_points);

    std::vector<Cluster> all_clusters = kMeans.getClusters();

    int rows = all_clusters.size();
    int cols = all_clusters[0].getPoint(0).getDimensions();
/*    auto** result_array = new double* [rows];

    for(int i = 0; i < rows; i++){
        result_array[i] = new double[cols];
        for(int j = 0; j < cols; j++){
            result_array[i][j] = all_clusters[i].getCentroidByPos(j);
        }
    }

    all_clusters = std::vector<Cluster>(); //Destroy vector
    return result_array;*/
    cpt = 0;
    auto result_array = new double[rows * cols];
    for(int i = 0; i < rows; i++){
       for(int j = 0; j < cols; j++){
           result_array[cpt] = all_clusters[i].getCentroidByPos(j);
           std::cout << result_array[cpt] << std::endl;
           cpt++;
       }
    }

    return result_array;
}


DLLEXPORT void print_array(double *v, size_t n) {
    for (size_t i = 0; i < n; i++) {
        printf("%f ", v[i]);
    }
    printf("\n");
}

DLLEXPORT void print_matrix(double *v, size_t n, size_t p) {
    std::cout << "Testing, " << v[0] << std::endl;
    std::cout << n << std::endl;
    std::cout << p << std::endl;
    Eigen::MatrixXd t(n, p);
    int cpt = 0;
    for (size_t i = 0; i < n; i++) { //4
        for (size_t j = 0; j < p; j++) { //2
            t((int)i, (int)j) = v[cpt];
            cpt++;
            printf("%f ", t((int)i, (int)j));
        }
        printf("\n");
    }
    printf("\n");
}
}