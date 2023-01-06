#pragma once
#include <iostream>
#include "models/ddpm.h"
class Trainer
{
public:
	Trainer(DDPM& ddpm,
		std::tuple<int, int> img_size,
		std::string &exp_name,
		int train_batch_size = 32,
		double train_lr = 2e-4,
		int train_num_epochs = 10000,
		double ema_decay = 0.995,
		int num_workers = 2,
		int save_and_sample_every = 100,
		int accumulation_steps = 2);
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> prepare_data(torch::Tensor x_real);
	template <typename DataLoader>
	void train(DataLoader& dataloader);
	void train(std::string dataset_path);
	torch::Device get_device();
private:
	DDPM ddpm{ nullptr };
	DDPM ddpm_shadow{ nullptr };
	std::tuple<int, int> img_size;
	int train_batch_size;
	double train_lr;
	int train_num_epochs;
	double ema_decay;
	int num_workers;
	int save_and_sample_every;
	int accumulation_steps;
	std::string sample_path;
	std::string checkpoint_path;
	torch::Device device = torch::Device(torch::kCPU);
	std::vector<int> steps;
};

