#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <random>
#include "tfdecoder.h"

using namespace std;
using namespace tf;

int tacotron_test(TFModel* tfmodel);

int main(int argc, char** argv) {
    std::string model_path = argv[1];
    TFModel* tfmodel = new TFModel();
    tfmodel->Init(model_path);
    time_t a = time(NULL);
    tacotron_test(tfmodel);
    time_t b = time(NULL);
    cout << "use time: " << (b-a) << endl;
    delete tfmodel;
    return 0;
}

int tacotron_test(TFModel* tfmodel) {
    std::string input_tensor_name = "inputs:0";
    std::string input_seq_name = "input_lengths:0";
    std::string output_tensor_name = "model/griffinlim/Squeeze:0";
    std::vector<std::string> output_tensor_names;
    output_tensor_names.push_back(output_tensor_name);
    int rows = 48;
    int cols = 132;
    FILE *fin = fopen("input.bin", "rb");
    std::vector<float> input_datas;
    for (int i = 0; i < rows; i++) {
        std::vector<float> vec;
        for (int j = 0; j < cols; j++) {
            float temp = -1;
            fread(&temp, sizeof(float), 1, fin);
            input_datas.push_back(temp);
        }
    }
    fclose(fin);
    std::vector<long long int> input_dims;
    input_dims.push_back(1);
    input_dims.push_back(rows);
    input_dims.push_back(cols);
    std::vector<long long int> len_dims;
    len_dims.push_back(1);

    std::vector<int> len_datas;
    len_datas.push_back(rows);

    LayerData_v3 input_feats(input_tensor_name, input_dims, input_datas.data(), 0);
    LayerData_v3 input_len(input_seq_name, len_dims, len_datas.data(), 1);
    std::vector<LayerData_v3> all_in = {input_feats, input_len};

    std::vector<std::vector<float> >  output_values;
    if (!tfmodel->Evaluate_v4(&all_in, output_tensor_names, output_values)) {
        std::cout << "predict ok" << std::endl;
    }
}
