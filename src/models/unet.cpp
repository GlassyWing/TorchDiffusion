#include "unet.h"
#include "../utils/hand.h"

GroupNormCustomImpl::GroupNormCustomImpl(int n_groups, int num_channels, float eps, bool affine) {
	this->n_groups = n_groups;
	this->num_channels = num_channels;
	this->eps = eps;
	this->affine = affine;

	if (affine) {
		weight = register_parameter("weight", torch::empty(num_channels));
		bias = register_parameter("bias", torch::empty(num_channels));
	}

	reset_paramters();
}

void GroupNormCustomImpl::reset_paramters() {
	if (affine) {
		torch::nn::init::ones_(weight);
		torch::nn::init::zeros_(bias);
	}
}

torch::Tensor GroupNormCustomImpl::forward(torch::Tensor x) {
	int b = x.size(0), c = x.size(1), h = x.size(2), w = x.size(3); // b, c, h, w
	x = x.permute({ 0, 2, 3, 1 });
	// (b, h, w, g, f) s.t. c = g * f
	x = x.reshape({ b, h, w, n_groups, c / n_groups });
	torch::Tensor var, mean;
	std::tie(var, mean) = torch::var_mean(x, {1, 2, 3}, true, true);
	auto norm_x = x.sub(mean).mul(torch::rsqrt(var.add(eps)));
	norm_x = norm_x.flatten(-2);
	if (affine) {
		norm_x = norm_x.mul(weight.reshape({ 1, 1, 1, -1 })).add(bias.reshape({ 1, 1, 1, -1 }));
	}
	return norm_x.permute({ 0, 3, 1, 2 });
}

torch::nn::Sequential Upsample(int dim, int dim_out) {

	torch::nn::UpsampleOptions options = torch::nn::UpsampleOptions()
		.scale_factor(std::vector<double>({ 2, 2 }))
		.mode(torch::kNearest);
	torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(dim, default_value(dim_out, dim), {3, 3})
		.padding(1)
		.bias(false);
	torch::nn::Sequential seq(
		torch::nn::Upsample(options),
		torch::nn::Conv2d(conv_options),
		torch::nn::SiLU(),
		GroupNormCustom(32, default_value(dim_out, dim))
	);
	
	return seq;
}

torch::nn::Sequential Downsample(int dim, int dim_out) {
	
	torch::nn::Conv2dOptions conv_options =
		torch::nn::Conv2dOptions(dim, default_value(dim_out, dim), { 3, 3 }).padding(1).bias(false);

	torch::nn::Sequential seq(
		torch::nn::Conv2d(conv_options),
		torch::nn::SiLU(),
		GroupNormCustom(32, default_value(dim_out, dim)),
		torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions({ 2, 2 }).stride({2, 2 }))
	);

	return seq;
}

ResidualBlockImpl::ResidualBlockImpl(int in_c, int out_c, int emb_dim, int n_groups) {
	this->out_c = out_c;
	conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_c, out_c, { 1, 1 }).bias(false));
	dense = torch::nn::Linear(torch::nn::LinearOptions(emb_dim, out_c).bias(false));
	fn1->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_c, out_c, { 3, 3 }).padding(1).bias(false)));
	fn1->push_back(torch::nn::SiLU());
	fn2->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(out_c, out_c, { 3, 3 }).padding(1).bias(false)));
	fn2->push_back(torch::nn::SiLU());
	pre_norm = GroupNormCustom(n_groups, out_c);
	post_norm = GroupNormCustom(n_groups, out_c);

	register_module("conv", conv);
	register_module("dense", dense);
	register_module("fn1", fn1);
	register_module("fn2", fn2);
	register_module("pre_norm", pre_norm);
	register_module("post_norm", post_norm);
}

torch::Tensor ResidualBlockImpl::forward(torch::Tensor x, torch::Tensor t) {
	torch::Tensor xi;
	if (x.size(1) == out_c) {
		xi = x.clone();
	}
	else {
		x = conv(x);
		xi = x.clone();
	}

	x = pre_norm(x);
	x = fn1->forward(x);
	x = x + dense(t).unsqueeze(-1).unsqueeze(-1);
	x = post_norm(x);
	x = fn2->forward(x);

	return xi + x;
}


void UnetImpl::init(int img_c, std::tuple<int, int>& img_size, std::vector<int>& scales, int emb_dim, int min_pixel, int n_block, int n_groups) {
	this->img_c = img_c;
	this->n_groups = n_groups;
	this->n_block = n_block;
	this->scales = scales;
	this->img_size = img_size;
	this->emb_dim = emb_dim;
	this->min_pixel = min_pixel;

	int img_height, img_width;
	std::tie(img_height, img_width) = img_size;
	min_img_size = std::min(img_height, img_width);

	stem = torch::nn::Conv2d(torch::nn::Conv2dOptions(img_c, emb_dim, { 3, 3 }).padding(1));
	auto skip_pooling = 0;
	auto cur_c = emb_dim;

	torch::OrderedDict<std::string, std::shared_ptr<Module>> enc_blocks;

	std::vector<std::tuple<int, int>> chs;

	// add enc blocks
	for (size_t i = 0; i < scales.size(); i++)
	{
		auto scale = scales[i];

		// sevaral residual blocks
		for (size_t j = 0; j < n_block; j++)
		{
			chs.push_back(std::make_tuple(cur_c, scale * emb_dim));
			auto block = ResidualBlock(cur_c, scale * emb_dim, emb_dim, n_groups);
			cur_c = scale * emb_dim;
			enc_blocks.insert((std::stringstream() << "enc_block_" << i * n_block + j).str(), block.ptr());
		}

		// downsample block if not reach to `min_pixel`.
		if (min_img_size > min_pixel) {
			enc_blocks.insert((std::stringstream() << "down_block_" << i).str(), Downsample(cur_c).ptr());
			min_img_size = min_img_size / 2;
		}
		else {
			skip_pooling += 1; // log how many times skip pooling.
		}
	}

	// add mid blocks
	enc_blocks.insert((std::stringstream() << "enc_block_" << scales.size() * n_block).str(),
		ResidualBlock(cur_c, cur_c, emb_dim, n_groups).ptr());
	this->encoder_blocks = torch::nn::ModuleDict(enc_blocks);

	std::reverse(chs.begin(), chs.end()); // decoder chs reversed.

	torch::OrderedDict<std::string, std::shared_ptr<Module>> dec_blocks;

	// add dec blocks, in reverse scales.
	size_t m = 0;
	for (int i = scales.size() - 1; i > -1; i--)
	{
		auto rev_scale = scales[i];
		if (m >= skip_pooling) {
			dec_blocks.insert((std::stringstream() << "up_block_" << m).str(), Upsample(cur_c).ptr());
		}

		for (size_t j = 0; j < n_block; j++)
		{
			int out_channels;
			int in_channels;
			std::tie(out_channels, in_channels) = chs[m * n_block + j];
			dec_blocks.insert((std::stringstream() << "dec_block_" << m * n_block + j).str(),
				ResidualBlock(in_channels, out_channels, emb_dim, n_groups).ptr());
			cur_c = out_channels;
		}
		m++;
	}

	// finaly add a to_rgb block
	torch::nn::Sequential to_rgb(
		GroupNormCustom(n_groups, cur_c),
		torch::nn::SiLU(),
		torch::nn::Conv2d(torch::nn::Conv2dOptions(cur_c, img_c, { 3, 3 }).padding(1).bias(false))
	);

	dec_blocks.insert(std::string("to_rgb"), to_rgb.ptr());
	this->decoder_blocks = torch::nn::ModuleDict(dec_blocks);

	stem = register_module("stem", stem);
	encoder_blocks = register_module("encoder_blocks", encoder_blocks);
	decoder_blocks = register_module("decoder_blocks", decoder_blocks);
}

UnetImpl::UnetImpl(int img_c, std::tuple<int, int>& img_size, std::vector<int>& scales, int emb_dim, int min_pixel, int n_block, int n_groups) {
	init(img_c, img_size, scales, emb_dim, min_pixel, n_block, n_groups);
}

UnetImpl::UnetImpl(UnetImpl& other) {
	init(other.img_c, other.img_size, other.scales, other.emb_dim, other.min_pixel, other.n_block, other.n_groups);
}

torch::Tensor UnetImpl::forward(torch::Tensor x, torch::Tensor t) {
	x = stem(x);

	std::vector<torch::Tensor> inners;

	inners.push_back(x);
	for (auto item : encoder_blocks->items()) {
		auto name = item.first;
		auto module = item.second;
		// resudial block
		if (startswith(name, "enc")) {
			x = module->as<ResidualBlock>()->forward(x, t);
			inners.push_back(x);
		}
		// downsample block
		else {
			x = module->as<torch::nn::Sequential>()->forward(x);
			inners.push_back(x);
		}
	}

	// drop last two (contains mid block output)
	auto inners_ = std::vector<torch::Tensor>(inners.begin(), inners.end() - 2);

	for (auto item : decoder_blocks->items()) {
		auto name = item.first;
		auto module = item.second;

		// upsample block
		if (startswith(name, "up")) {
			x = module->as<torch::nn::Sequential>()->forward(x);
			torch::Tensor xi = inners_.back(); inners_.pop_back(); // pop()
			x = x + xi;
		}
		// resudial block
		else if (startswith(name, "dec")) {
			torch::Tensor xi = inners_.back(); inners_.pop_back(); // pop()
			x = module->as<ResidualBlock>()->forward(x, t);
			x = x + xi;
		}
		else {
			x = module->as<torch::nn::Sequential>()->forward(x);
		}
	}

	return x;
}