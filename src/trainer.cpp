#include "trainer.h"
#include "utils/ema.h"
#include "utils/readfile.h"
#include "utils/path.h"
#include "datasets/folder.h"
#include <random>

Trainer::Trainer(DDPM& ddpm,
	std::tuple<int, int> img_size,
	std::string &exp_name,
	int train_batch_size,
	double train_lr,
	int train_num_epochs,
	double ema_decay,
	int num_workers,
	int save_and_sample_every,
	int accumulation_steps) {

	this->ddpm = ddpm;
	this->img_size = img_size;
	this->train_batch_size = train_batch_size;
	this->train_lr = train_lr * accumulation_steps; // auto scale lr by accumulation_steps
	this->train_num_epochs = train_num_epochs;
	this->ema_decay = ema_decay;
	this->num_workers = num_workers;
	this->save_and_sample_every = save_and_sample_every;
	this->accumulation_steps = accumulation_steps;

	sample_path = std::string("experiments").append({ file_sepator() }).append(exp_name).append({ file_sepator() }).append("outputs");
	checkpoint_path = std::string("experiments").append({ file_sepator() }).append(exp_name).append({ file_sepator() }).append("checkpoints");

	// make experiments save path
	makedirs(checkpoint_path.c_str());
	makedirs(sample_path.c_str());

	// Get model device.
	device = ddpm->parameters(true).back().device();
	
	// make a new ddpm as shadow.
	ddpm_shadow = DDPM(Unet(*(ddpm->get_model()->as<Unet>())).ptr(), img_size, ddpm->T, ddpm->embedding_size);
	ddpm_shadow->to(device);
	
	// Init copy
	update_average(ddpm_shadow.ptr(), ddpm.ptr(), 0.);

	for (int i = 0; i < ddpm->T; i++)
	{
		steps.push_back(i);
	}
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Trainer::prepare_data(torch::Tensor x_real) {
	auto batch_images = x_real.to(device);
	auto batch_size = batch_images.size(0);

	std::vector<int> steps;
	// sample batch size steps from 0...T
	/*std::random_device rd;
	std::default_random_engine generator(rd());
	std::uniform_int_distribution<int> distribution(0, ddpm->T - 1);

	
	for (int i = 0; i < batch_size; i++)
	{
		steps.push_back(distribution(generator));
	}*/

	std::random_shuffle(this->steps.begin(), this->steps.end());
	for (int i = 0; i < batch_size; i++)
	{
		steps.push_back(this->steps.at(i));
	}

	auto batch_steps = torch::tensor(steps, torch::TensorOptions().device(device).dtype(torch::kLong));
	auto batch_bar_alpha = (ddpm->bar_alpha).index({ batch_steps }).to(torch::kFloat).reshape({ -1, 1, 1, 1 });
	auto batch_bar_beta = (ddpm->bar_beta).index({ batch_steps }).to(torch::kFloat).reshape({ -1, 1, 1, 1 });

	auto batch_noise = torch::randn_like(batch_images);
	auto batch_noise_images = batch_images * batch_bar_alpha + batch_noise * batch_bar_beta;

	return std::make_tuple(batch_noise_images, batch_steps, batch_noise);
}

void Trainer::train(std::string dataset_path) {
	auto dataset = ImageFolderDataset(dataset_path, img_size)
		// RandomFliplr
		.map(torch::data::transforms::Lambda<torch::data::TensorExample>([](torch::data::TensorExample input) {
				if ((torch::rand(1).item<double>() < 0.5)) {
					input.data = torch::flip(input.data, {-1});
				}
				return input;
			}))
		.map(torch::data::transforms::Stack<torch::data::TensorExample>());

	const size_t dataset_size = dataset.size().value();
	const size_t num_step = dataset_size / train_batch_size;

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(dataset), 
		torch::data::DataLoaderOptions()
			.batch_size(train_batch_size)
			.workers(num_workers)
			.drop_last(true)
		);
	
	torch::optim::AdamW optimizer(ddpm->parameters(), train_lr);

	long step = 0;
	for (size_t epoch = 1; epoch <= train_num_epochs; ++epoch) {
		
		size_t batch_idx = 0;
		ddpm->zero_grad(true);
		for (auto &batch : *train_loader) {
			auto data = batch.data.to(device);

			torch::Tensor noise_images;
			torch::Tensor steps;
			torch::Tensor noise;
			std::tie(noise_images, steps, noise) = prepare_data(data);

			auto denoise = ddpm->forward(noise_images, steps);
			auto loss = torch::sum((denoise - noise).pow(2), { 1, 2, 3 }, true).mean();
			loss = loss / accumulation_steps;

			loss.backward();

			if ((step + 1) % accumulation_steps == 0) {
				optimizer.step();
				ddpm->zero_grad(true);
				update_average(ddpm_shadow.ptr(), ddpm.ptr(), ema_decay);
			}

			std::printf("\rEpoch [%1zd / %5d] .. [%5zd/%5zd] Loss: %.4f",
				epoch,
				train_num_epochs,
				batch_idx,
				num_step,
				loss.template item<float>());

			step += 1;
			batch_idx += 1;

			if ((step != 0) && (step % save_and_sample_every == 0)) {
				auto exp_img_path = (std::stringstream() << sample_path << "/ddpm_ckpt_" << epoch << "_" << step << "_ema.png").str();
				ddpm_shadow->sample(exp_img_path, 4);
				torch::save(ddpm_shadow, (std::stringstream() << checkpoint_path << "/ddpm_ckpt_" << epoch << "_" << step << ".pth").str());
			}
		}
	}
}