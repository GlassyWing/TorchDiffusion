#include <iostream>
#include <torch/torch.h>

#include "unet.h"



class DDPMImpl : public torch::nn::Module {
private:
	std::tuple<int, int> img_size;
	
	std::shared_ptr<Module> model{ nullptr };
public:
	int T;
	torch::Tensor alpha;
	torch::Tensor beta;
	torch::Tensor sigma;
	torch::Tensor bar_alpha;
	torch::Tensor bar_beta;
	torch::Tensor t;
	int embedding_size;

	DDPMImpl(const std::shared_ptr<Module>& model, std::tuple<int, int>& img_size, int T = 1000, int embedding_size = 2);
	torch::Tensor forward(torch::Tensor x, torch::Tensor t_in);
	torch::Tensor sample(std::shared_ptr<std::string> path = nullptr, int n = 4, std::shared_ptr<torch::Tensor> z_samples = nullptr,
		int t0 = 0, torch::Device device = torch::Device(torch::kCUDA, 0));
	torch::Tensor sample(std::string path, torch::Device device = torch::Device(torch::kCUDA, 0));
	torch::Tensor sample(std::string path, int n, torch::Device device = torch::Device(torch::kCUDA, 0));
	torch::Tensor sample(torch::Device device = torch::Device(torch::kCUDA, 0));
	torch::Tensor ddim_sample(std::shared_ptr<std::string> path = nullptr, int n = 4, std::shared_ptr<torch::Tensor> z_samples = nullptr,
		int stride = 1, float eta = 1, torch::Device device = torch::Device(torch::kCUDA, 0));

	std::shared_ptr<Module> get_model();
}; TORCH_MODULE(DDPM);