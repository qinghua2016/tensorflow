#include "tfdecoder.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include <iostream>
#include <string.h>
using namespace tensorflow;
using namespace std;

namespace tf {

LayerData_v3::LayerData_v3(const std::string& name, std::vector<long long int>& dims, void* data, int type) {
    this->name = name;
    this->dims = dims;
    this->data_ = data;
    this->type_ = type;
}

LayerData_v3::~LayerData_v3() {

}

TFModel::TFModel() {
    session = nullptr;
}

bool TFModel::Init(const std::string& model_file, int intra_thread, int inter_thread) {
    this->empty.clear();
    ConfigProto config = ConfigProto();
    SessionOptions sessOpt = SessionOptions();
    config.set_intra_op_parallelism_threads(intra_thread);
    config.set_inter_op_parallelism_threads(inter_thread);
    sessOpt.config = config;
    Status status = NewSession(sessOpt, &session);
    if (!status.ok()) {
        std::cout << "create session wrong" << std::endl;
        return false;
    }
    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), model_file, &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return false;
    }
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return false;
    }
    return true;
}

bool TFModel::Init(const char* buffer, int len, int intra_thread, int inter_thread) {
    this->empty.clear();
    ConfigProto config = ConfigProto();
    SessionOptions sessOpt = SessionOptions();
    config.set_intra_op_parallelism_threads(intra_thread);
    config.set_inter_op_parallelism_threads(inter_thread);
    config.mutable_gpu_options()->set_visible_device_list("0");
    config.mutable_gpu_options()->set_allow_growth(false);
    config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.3);
    sessOpt.config = config;
    Status status = NewSession(sessOpt, &session);
    if (!status.ok()) {
        std::cout << "create session wrong" << std::endl;
        return false;
    }
    GraphDef graph_def;
    status = ReadBinaryProtoFromBuffer(Env::Default(), buffer, len, &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return false;
    }
    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return false;
    }
    return true;
}

TFModel::~TFModel() {
    if (session) {
        session->Close();
        delete session;
    }
}

bool TFModel::Evaluate_v4(std::vector<LayerData_v3>* input_layers, std::vector<std::string> output_node_name, std::vector<std::vector<float> >& output_values) {
    std::vector<std::pair<std::string, Tensor> > inputs;
    for (auto& layer : (*input_layers)) {
        long long int in_len_all = 1;
        for (auto& dim : layer.dims) {
            in_len_all *= dim;
        }

        if (layer.type_ == 0) {
            Tensor input(DT_FLOAT, TensorShape(layer.dims));
            memcpy(input.flat<float>().data(), layer.data_, in_len_all*sizeof(float));
            inputs.push_back({layer.name, input});
        } else if (layer.type_ == 1) {
            Tensor input(DT_INT32, TensorShape(layer.dims));
            memcpy(input.flat<int>().data(), layer.data_, in_len_all*sizeof(int));
            inputs.push_back({layer.name, input});
        }
    }

    std::vector<Tensor> outputs;
    Status status = ((Session*)session)->Run(inputs, {output_node_name}, {}, &outputs);
    if (!status.ok()) {
        std::cerr << status.ToString().c_str() << std::endl;
        return false;
    }
    for (size_t k = 0; k < outputs.size(); k++) {
        Tensor t = outputs[k];
        int dims = t.dims();
        int output_len = 1;
        for (int i = 0; i < dims; i++) {
            output_len = output_len * t.dim_size(i);
        }
        std::vector<float> output_value;
        int dtype = t.dtype();
        if (dtype == 3) {
            auto ttmp = t.flat<int>();
            for (int i = 0; i < output_len; i++) {
                output_value.push_back(ttmp(i));
            }
        } else if (dtype == 1) {
            auto ttmp = t.flat<float>();
            for (int i = 0; i < output_len; i++) {
                output_value.push_back(ttmp(i));
            }
        }
        output_values.push_back(output_value);
    }
    return true;
}

}
