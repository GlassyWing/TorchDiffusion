#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "src/trainer.h"
#include "src/datasets/folder.h"
#include "src/utils/weight.h"
#include "src/utils/model_serialize.h"
#include "src/utils/hand.h"
#include "CLI11.hpp"

using namespace cv;

// this function use to test the Unet model's validity.
void test_unet_train(Unet& model, 
	std::string dataset_path, 
	std::tuple<int, int> img_size, 
	int train_batch_size, 
	int num_workers, 
	int epochs=100,
	torch::Device device = torch::Device(torch::kCPU)) {
	auto dataset = ImageFolderDataset(dataset_path, img_size)
		.map(torch::data::transforms::Stack<torch::data::TensorExample>());

	const size_t dataset_size = dataset.size().value();
	const size_t num_step = dataset_size / train_batch_size;

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(dataset),
		torch::data::DataLoaderOptions()
		.drop_last(true)
		.batch_size(train_batch_size)
		.workers(num_workers));

	torch::optim::AdamW optimizer(model->parameters(), 2e-4);

	auto step = 0;
	for (size_t epoch = 1; epoch <= epochs; ++epoch) {
		model->train();
		size_t batch_idx = 0;

		for (auto& batch : *train_loader) {
			auto images = batch.data.to(device);
			auto steps = torch::zeros({ images.size(0), 64 }, torch::TensorOptions().dtype(images.dtype()).device(device));

			optimizer.zero_grad();
			auto denoise = model->forward(images + 0.5 * torch::randn_like(images), steps);
			auto loss = torch::sum((denoise - images).pow(2), { 1, 2, 3 }, true).mean();

			loss.backward();
			optimizer.step();

			std::printf("\rEpoch [%1zd / %5d] .. [%5zd/%5zd] Loss: %.4f",
				epoch,
				epochs,
				batch_idx,
				num_step,
				loss.template item<float>());
			batch_idx += 1;

			if ((step + 1) % 50 == 0) {
				auto t = denoise.detach().cpu().index({0}); // (h, w, 3)
				t = t.permute({ 1, 2, 0 }).contiguous();
				t = (t + 1) / 2. * 255;
				t = t.to(torch::kU8);
				cv::Mat mat = cv::Mat(128, 128, CV_8UC3, t.data_ptr());
				cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
				cv::imwrite((std::stringstream() << "test_" << step << ".png").str(), mat);
			}
			
			step += 1;
		}
	}
}

// this function use to test whether model's output and grad valid.
void test_out_grad() {

	torch::manual_seed(0);
	
	auto inp = torch::randn({ 2, 3, 8, 8 }).set_requires_grad(true);
	auto te = torch::zeros({ 2 }, torch::TensorOptions().dtype(torch::kLong));

	auto unet = Unet(3, std::make_tuple<>(8, 8), std::vector<int>({ 1, 1 }), 64, 1);
	auto ddpm = DDPM(unet.ptr(), std::make_tuple<>(8, 8), 10, 64);
	auto model = ddpm;

	try {
		auto out = model->forward(inp, te);
		auto loss = (torch::ones_like(out) - out).pow(2).mean();
		loss.backward();

		std::cout << "-------------------------------" << std::endl;
		std::cout << inp.index({ 0, 0 }) << std::endl;
		std::cout << out.sizes() << " " << out.index({ 0, 0 }) << std::endl;
		std::cout << inp.grad().index({ 0, 0 }) << std::endl;

		auto ddpm_out = model->sample(torch::Device(torch::kCPU));
		std::cout << ddpm_out.index({ 0, 0 }) << std::endl;
	}
	catch (const std::exception& e) {
		std::cout << e.what();
		
	}
}

// test dataset.get image
void test_dataset_load(std::string dataset_path, std::tuple<int, int> img_size) {
	auto dataset = ImageFolderDataset(dataset_path, img_size);

	auto data = dataset.get(0).data;
	std::cout << &data << " " << data.data_ptr() << std::endl;
	auto data2 = dataset.get(1).data;
	std::cout << &data2 << " " << data2.data_ptr() << std::endl;
	std::cout << (data == data2).all() << std::endl;
	std::cout << (data.data_ptr() == data2.data_ptr()) << std::endl;

	auto t = torch::stack({data, data2}, 0);

	t = t.index({0}); // (h, w, 3)
	t = t.permute({ 1, 2, 0 }).contiguous();
	t = (t + 1) / 2. * 255;
	t = t.to(torch::kU8);
	cv::Mat mat = cv::Mat(128, 128, CV_8UC3, t.data_ptr());
	cv::imwrite("./test_2.png", mat);
}

auto main(int argc, char** argv) -> int {
	
	at::globalContext().setBenchmarkCuDNN(true);

	/*------------define model scale.------------*/
	std::tuple<int, int> img_size({ 128, 128 });	// image size (height, width)
	std::vector<int> scales = { 1 ,1 , 2, 2, 4, 4 };// scale size for each block, base on emb_dim
	int emb_dim = 64;								// base channels dim
	int T = 1000;									// ddpm steps
	auto device = torch::Device(torch::kCUDA, 0);	// indicate training/testing on which device
	/*-------------------------------------------*/
	
	/*------------define train/test params--------*/
	std::string mode("train");
	std::string exp_name;
	std::string dataset_path(R"(D:\datasets\thirds\anime_face)");
	std::string pretrained_weights;
	std::string weight_path(R"(D:\projects\personal\ddpm\demo.pth)");
	int batch_size = 32;
	double learning_rate = 2e-4;
	int num_epochs = 1000;
	double ema_decay = 0.995;
	int num_workers = 4;
	int save_and_sample_every = 500;
	int accumulation_steps = 2;
	/*--------------------------------------------*/

	CLI::App app{ "TorchLib Diffusion model implementation" };
	app.add_option("-m,--mode", mode, "train/test. Default. train");
	app.add_option("-d,--dataset", dataset_path, "dataset path");
	app.add_option("-b,--batch_size", batch_size, "batch size");
	app.add_option("--lr,--learning_rate", learning_rate, "learning rate");
	app.add_option("--epochs", num_epochs, "number of epochs.");
	app.add_option("--n_cpu", num_workers, "number of cpu threads to use during batch generation");
	app.add_option("--name", exp_name, "experiment name");
	app.add_option("-p,--pretrained_weights", pretrained_weights, "pretrained model path");
	app.add_option("-w,--weight_path", weight_path, "weight_path for inference");
	CLI11_PARSE(app, argc, argv);
	

	auto model = Unet(3, img_size, scales, emb_dim);
	auto diffusion = DDPM(model.ptr(), img_size, T, emb_dim);

	if (mode == "test") {
		std::cout << "Running at inference mode..." << std::endl;
		try {
			// load pytorch trained model.
			if (endswith(weight_path, "pth")) {
				load_state_dict(diffusion.ptr(), weight_path);
			}
			else {
				torch::load(diffusion, weight_path);
			}
			std::cout << "Load weight successful!" << std::endl;
			
			diffusion->to(device);
			diffusion->eval();

			int64 start = cv::getTickCount();
			auto x_samples = diffusion->sample("./demo.png", device);
			double duration = (cv::getTickCount() - start) / cv::getTickFrequency();
			std::cout << duration / T << " s per prediction" << std::endl;
		}
		catch (const std::exception& e) {
			std::cout << e.what() << std::endl;
			return -1;
		}
	}
	else {
        try {
            std::cout << "Running at training mode..." << std::endl;
            std::cout << "Experiments name: " << (exp_name.empty() ? "(Empty)" : exp_name) << std::endl;
            diffusion->apply(weights_norm_init());

            if (!pretrained_weights.empty()) {
                if (endswith(pretrained_weights, "pth")) {
                    load_state_dict(diffusion.ptr(), pretrained_weights);
                }
                else {
                    torch::load(diffusion, pretrained_weights);
                }
                std::cout << "Load pretrained weight " << pretrained_weights << " successful!" << std::endl;
            }
            diffusion->to(device);
            auto trainer = Trainer(diffusion, img_size,
                    /*exp_name=*/exp_name,
                    /*train_batch_size=*/batch_size,
                    /*train_lr=*/learning_rate,
                    /*train_num_epochs=*/num_epochs,
                    /*ema_decay=*/ema_decay,
                    /*num_workers=*/num_workers,
                    /*save_and_sample_every=*/save_and_sample_every,
                    /*accumulation_steps=*/accumulation_steps
            );
            trainer.train(dataset_path);
        } catch (std::exception &e) {
            std::cout << e.what() << std::endl;
        }

	}
	
	return 0;
}
