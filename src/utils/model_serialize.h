#pragma once
#include <torch/torch.h>
std::vector<char> get_the_bytes(std::string filename);

void load_state_dict(std::shared_ptr<torch::nn::Module> model, std::string pt_pth);