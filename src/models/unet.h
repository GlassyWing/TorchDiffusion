#include <torch/torch.h>

// A custom group norm implementation.
class GroupNormCustomImpl : public torch::nn::Module {
public:
	GroupNormCustomImpl(int n_groups, int num_channels, float eps = 1e-6, bool affine = true);
	void reset_paramters();
	torch::Tensor forward(torch::Tensor x);
private:
	int n_groups;
	int num_channels;
	float eps;
	bool affine;
	torch::Tensor weight;
	torch::Tensor bias;
}; TORCH_MODULE(GroupNormCustom);

// Create a 2x Upsample block, if `dim_out` less than 0, the output dim will be `dim`
torch::nn::Sequential Upsample(int dim, int dim_out = -1);

// Create a 2x Downsample block, if `dim_out` less than 0, the output dim will be `dim`
torch::nn::Sequential Downsample(int dim, int dim_out = -1);

// A resudual block impl, support pass `t` emb.
class ResidualBlockImpl : public torch::nn::Module {
public:
	ResidualBlockImpl(int in_c, int out_c, int emb_dim, int n_groups = 32);
	torch::Tensor forward(torch::Tensor x, torch::Tensor t);
private:
	torch::nn::Conv2d conv{ nullptr };
	torch::nn::Linear dense{ nullptr };
	torch::nn::Sequential fn1;
	torch::nn::Sequential fn2;
	int out_c;
	GroupNormCustom pre_norm{ nullptr };
	GroupNormCustom post_norm{ nullptr };
}; TORCH_MODULE(ResidualBlock);

class UnetImpl : public torch::nn::Module {
public:
	UnetImpl(int img_c, std::tuple<int, int> &img_size, std::vector<int>& scales, int emb_dim, int min_pixel = 4, int n_block = 2, int n_groups=32);
	UnetImpl(UnetImpl& other);
	
	torch::Tensor forward(torch::Tensor x, torch::Tensor t);
private:
	int n_block;	// each block contains `n_block * (ResdualBlock)`
	int img_c;
	std::vector<int> scales; // save channel dim for each block.
	std::tuple<int, int> img_size;	// model input size.
	int n_groups;	// global GroupNorm param.
	int min_img_size; // min img_size
	int emb_dim;
	int min_pixel;

	torch::nn::Conv2d stem{ nullptr };	// init feature extractor.
	torch::nn::ModuleDict encoder_blocks{ nullptr };
	torch::nn::ModuleDict decoder_blocks{ nullptr };

	void init(int img_c, std::tuple<int, int>& img_size, std::vector<int>& scales, int emb_dim, int min_pixel = 4, int n_block = 2, int n_groups = 32);
}; TORCH_MODULE(Unet);