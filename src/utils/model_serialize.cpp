#include "model_serialize.h"

std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
            (std::istreambuf_iterator<char>(input)),
            (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}


void load_state_dict(std::shared_ptr<torch::nn::Module> model, std::string pt_pth) {
    std::vector<char> f = get_the_bytes(pt_pth);

    c10::Dict<torch::IValue, torch::IValue> weights = torch::pickle_load(f).toGenericDict();

    const torch::OrderedDict<std::string, at::Tensor> &model_params = model->named_parameters();
    std::vector<std::string> param_names;
    for (auto const &w: model_params) {
        param_names.push_back(w.key());
    }

    torch::NoGradGuard no_grad;
    for (auto const &w: weights) {
        std::string name = w.key().toStringRef();
        at::Tensor param = w.value().toTensor();

        if (std::find(param_names.begin(), param_names.end(), name) != param_names.end()) {
            model_params.find(name)->copy_(param);
        } else {
            std::cout << name << " does not exist among model parameters." << std::endl;
        };

    }
}