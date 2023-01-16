//#pragma once
//#ifndef DIFFUSION_DDIM_H
//#define DIFFUSION_DDIM_H
//
//#include "sampler.h"
//
//
//class DDIMImpl : public DDPMImpl {
//
//public:
//    DDIMImpl(const std::shared_ptr<Module> &model, SamplerOptions &options);
//    torch::Tensor
//    sample(std::shared_ptr<std::string> path = nullptr, int n = 4, std::shared_ptr<torch::Tensor> z_samples = nullptr,
//           int t0 = 0, torch::Device device = torch::Device(torch::kCUDA, 0));
//
//    Sampler copy();
//
//    torch::Tensor sample(std::string path, torch::Device device = torch::Device(torch::kCUDA, 0));
//
//    torch::Tensor sample(std::string path, int n, torch::Device device = torch::Device(torch::kCUDA, 0));
//
//    torch::Tensor sample(torch::Device device = torch::Device(torch::kCUDA, 0));
//
//public:
//    torch::Tensor bar_alpha_;
//    torch::Tensor bar_alpha_pre_;
//    torch::Tensor bar_beta_;
//    torch::Tensor bar_beta_pre_;
//    torch::Tensor alpha_;
//    torch::Tensor sigma_;
//    torch::Tensor epsilon_;
//    int stride;
//};
//
//TORCH_MODULE(DDIM);
//
//
//#endif //DIFFUSION_DDIM_H
