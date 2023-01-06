#include "ddpm.h"
#include <opencv2/opencv.hpp>

DDPMImpl::DDPMImpl(std::shared_ptr<Module> model, std::tuple<int, int>& img_size, int T, int embedding_size) {
	this->model = model;
	this->img_size = img_size;
	this->T = T;
	this->embedding_size = embedding_size;

	alpha = register_buffer("alpha", torch::sqrt(1.0 - 0.02 * torch::arange(1, T + 1) / (double) T));
	beta = register_buffer("beta", torch::sqrt(1.0 - torch::pow(alpha, 2)));
	bar_alpha = register_buffer("bar_alpha", torch::cumprod(alpha, 0));
	bar_beta = register_buffer("bar_beta", torch::sqrt(1.0 - bar_alpha.pow(2)));
	sigma = register_buffer("sigma", beta.clone());
	t = register_buffer("t", positional_encoding_1d(embedding_size, T));

	register_module("model", model);
}

std::shared_ptr<torch::nn::Module> DDPMImpl::get_model() {
	return model;
}

torch::Tensor DDPMImpl::forward(torch::Tensor x, torch::Tensor t_in) {
	auto t_e = t.index({ t_in });
	auto x_r = model->as<Unet>()->forward(x, t_e);
	return x_r;
}
torch::Tensor DDPMImpl::sample(std::string path, torch::Device device) {
	return sample(std::make_shared<std::string>(path), 4, nullptr, 0, device);
}
torch::Tensor DDPMImpl::sample(std::string path, int n, torch::Device device) {
	return sample(std::make_shared<std::string>(path), n, nullptr, 0, device);
}

torch::Tensor DDPMImpl::sample(torch::Device device) {
	return sample(nullptr, 4, nullptr, 0, device);
}

torch::Tensor DDPMImpl::sample(std::shared_ptr<std::string> path, int n, std::shared_ptr<torch::Tensor> z_samples,
	int t0, torch::Device device) {
	torch::NoGradGuard no_grad;
	torch::Tensor z;

	int img_height, img_width;
	std::tie(img_height, img_width) = img_size;

	if (z_samples == nullptr) {
		z = torch::randn({ (long)std::pow(n, 2), 3, img_height, img_width }, torch::TensorOptions().device(device));
	}
	else {
		z = z_samples->clone();
	}

	for (size_t t = t0; t < T; t++)
	{
		int ti = T - t - 1;
		auto bti = torch::tensor({ti}, torch::TensorOptions().device(device)).repeat(z.size(0));
		z -=  beta.index({ ti }).pow(2) / bar_beta.index({ ti }) * forward(z, bti);
		z /=  alpha.index({ ti });
		z +=  torch::randn_like(z) * sigma.index({ ti });

		std::printf("\rSample [%1zd / %5d] ", t, T);
	}

	auto x_samples = torch::clip(z, -1, 1); // (n * n, 3, h, w)
	auto result = x_samples.clone();
	x_samples = ((x_samples + 1.) / 2. * 255).detach().to(torch::kU8).cpu().permute({ 0, 2, 3, 1 }).contiguous(); // (n * n, h, w, 3)
	if (path != nullptr) {
		
		auto figure = torch::zeros({ img_height * n, img_width * n, 3 }, torch::TensorOptions().dtype(torch::kU8));
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++) {
				auto digit = x_samples.index({ i * n + j });
				figure.index_put_({ Slice(i * img_height, (i + 1) * img_height), 
						Slice(j * img_width, (j + 1) * img_width) }, 
					digit);
			}
		}

		cv::Mat result_img(img_height * n, img_width * n, CV_8UC3, figure.data_ptr());
		cv::cvtColor(result_img, result_img, cv::COLOR_RGB2BGR);
		cv::imwrite(*path, result_img);
	}

	return result;
}