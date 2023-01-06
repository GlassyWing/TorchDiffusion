#include <iostream>
#include <torch/torch.h>

#include "unet.h"

using namespace torch::indexing;

// @param d_model:  dimension of the model
// @param length:	length of positions
// @ret: length * d_model position matrix
inline torch::Tensor positional_encoding_1d(int d_model, int length) {
	if ((d_model % 2) != 0) {
		throw (std::stringstream() << "Cannot use sin/cos positional encoding with odd dim (got dim=" << d_model << ")").str();
	} 

	auto pe = torch::zeros({length, d_model});
	auto position = torch::arange(0, length, torch::TensorOptions().dtype(torch::kFloat)).unsqueeze(1);
	auto div_term = torch::exp(torch::arange(0, d_model, 2, torch::TensorOptions().dtype(torch::kFloat)) * -(std::log(10000.0) / d_model));
	
	pe.index_put_({ Slice(), Slice(0, None, 2)}, torch::sin(position * div_term));
	pe.index_put_({ Slice(), Slice(1, None, 2)}, torch::cos(position * div_term));

	return pe;
}
	

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

	DDPMImpl(std::shared_ptr<Module> model, std::tuple<int, int>& img_size, int T = 1000, int embedding_size = 2);
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