#ifndef TF_DECODER_H
#define TF_DECODER_H

#include <string>
#include <vector>

namespace tensorflow {
    class Session;
    class Tensor;
}

namespace tf {

class LayerData_v2 {
public:
    LayerData_v2(const std::string& name, std::vector<long long int>& dims, float* data);
    std::string name;
    std::vector<long long int> dims;
    float* data;
};

class LayerData_v3 {
public:
    LayerData_v3(const std::string& name, std::vector<long long int>& dims, void* data, int type=0);
    ~LayerData_v3();
    std::string name;
    std::vector<long long int> dims;
    void * data_;
    int type_;
};

class TFModel {
public:
    TFModel();
    bool Init(const std::string& model_file, int intra_thread=2, int inter_thread=32);
    bool Init(const char* buffer, int len, int intra_thread=2, int inter_thread=32);
    ~TFModel();
    bool Evaluate_v4(std::vector<LayerData_v3>* input_layers, std::vector<std::string> output_node_names, std::vector<std::vector<float> >& output_values);
private:
    tensorflow::Session *session;
    std::vector<long long int> empty;
};

}

#endif
