#include "TComTorch.h"
#include <iostream>
#include <fstream>

#include "torch/torch.h"
#include "torch/script.h"
#include "math.h"

using namespace torch;

struct Tanh : public torch::nn::Module {
	Tanh() {}

	torch::Tensor forward(torch::Tensor x) {
		return torch::tanh(x);
	}
};

class Res : public nn::Module
{
public:
	nn::Sequential conv = nullptr;
	Res(int ch, int size, int group = 1) {
		nn::Conv2dOptions ConvOption(ch, ch, size);
		ConvOption.groups(group);
		nn::Conv2dOptions PaddingOption(ch, ch, 1);
		PaddingOption.padding((size - 1) >> 1).groups(ch).with_bias(false);
		conv = nn::Sequential(nn::Conv2d(ConvOption),
			Tanh(),
			nn::Conv2d(PaddingOption)
		);
		register_module("conv", conv);
	}
	auto forward(Tensor x) {
		return x + conv->forward(x);
	}
};

class Module : public nn::Module
{
public:
	nn::Sequential conv1 = nullptr;
	nn::Sequential conv2 = nullptr;
	bool gpu = false;

public:
	Module()
	{
		nn::Conv2dOptions ConvStrideOptions(2, 128, 10);
		nn::Conv2dOptions ConvTransposeOptions(64, 1, 10);
		ConvStrideOptions.stride(8);
		ConvTransposeOptions.transposed(true);
		ConvTransposeOptions.stride(8);

		conv1 = nn::Sequential(
			nn::Conv2d(ConvStrideOptions),
			nn::Conv2d(nn::Conv2dOptions(128, 128, 1)),
			nn::Conv2d(nn::Conv2dOptions(128, 64, 1)),
			nn::Conv2d(ConvTransposeOptions)
		);
		conv2 = nn::Sequential(
			nn::Conv2d(nn::Conv2dOptions(2, 64, 5)),
			Res(64, 3),
			Res(64, 3),
			nn::Conv2d(nn::Conv2dOptions(64, 128, 3)),
			Res(128, 3),
			nn::Conv2d(nn::Conv2dOptions(128, 64, 3)),
			Res(64, 3),
			Res(64, 3),
			nn::Conv2d(nn::Conv2dOptions(64, 1, 3)),
			Res(1, 3)
		);

		register_module("conv1", conv1);
		register_module("conv2", conv2);
	}

	Tensor forward(Tensor x)
	{
		static nn::Conv2d* p_paddingTranspose = NULL;
		if (p_paddingTranspose == NULL)
		{
			nn::Conv2dOptions PaddingTransposeOption(1, 1, 1);
			PaddingTransposeOption.transposed(true).padding(5).with_bias(false);
			p_paddingTranspose = new nn::Conv2d(PaddingTransposeOption);
			auto& paddingTranspose = *p_paddingTranspose;
			paddingTranspose->weight.set_requires_grad(false);
			paddingTranspose->weight[0][0][0][0] = 1.;
			if (gpu)
				paddingTranspose->to(kCUDA);
		}
		auto& paddingTranspose = *p_paddingTranspose;

		auto y = conv1->forward(x);
		y[0][0] = y[0][0] + x[0][0];
		auto ym = paddingTranspose->forward(y);
		auto z = x;
		z[0][0] = y[0][0];

		auto s = conv2->forward(z);

		return s + ym;
	}


};

std::ofstream fout;

void print(at::Tensor x)
{
	if (!(fout.is_open()))
		fout.open("pytorch.txt", fout.app);
	at::print(fout, x, 80i64);
	fout << std::endl;
}

void pic_cpy(float_t* dst, TComPicYuv* rec_pic, Rect patch, int stride_dst, ComponentID CompID = COMPONENT_Y)
{
	Pel* rec_pic_y = rec_pic->getAddr(CompID);
	int height = rec_pic->getHeight(CompID);
	int width = rec_pic->getWidth(CompID);
	int stride = rec_pic->getStride(CompID);
	// int stride_dst = patch.endX - patch.startX + 26;
	dst += 21 - patch.startX + (21 - patch.startY) * stride_dst;

	int i, j;
	for (i = ((patch.startY == 0) ? 0 : (patch.startY - 21)); i < ((patch.endY == height) ? height : (patch.endY + 21)); i++)
	{
		for (j = ((patch.startX == 0) ? 0 : (patch.startX - 21)); j < ((patch.endX == width) ? width : (patch.endX + 21)); j++)
			dst[i*stride_dst + j] = rec_pic_y[i*stride + j];

		if (patch.startX == 0)
			for (j = -21; j < 0; j++)
				dst[i*stride_dst + j] = rec_pic_y[i*stride + (-j - 1)];

		if (patch.endX == width)
			for (j = width; j < width + 21; j++)
				dst[i*stride_dst + j] = rec_pic_y[i*stride + (2 * width - j - 1)];
	}

	if (patch.startY == 0)
		for (i = -21; i < 0; i++)
		{
			for (j = patch.startX - 21; j < patch.endX + 21; j++)
				dst[i*stride_dst + j] = dst[(-i - 1)*stride_dst + j];
		}

	if (patch.endY == height)
		for (int i = height; i < height + 21; i++)
		{
			for (j = patch.startX - 21; j < patch.endX + 21; j++)
				dst[i*stride_dst + j] = dst[(2 * height - i - 1)*stride_dst + j];
		}
}

void pic_cpy(TComPic* pcPic, TComPicYuv* rec_pic, float_t* src, Rect patch, int stride_dst,  ComponentID CompID = COMPONENT_Y)
{
	//TComPicYuv* rec_pic = pcPic->getPicYuvRec();
	Pel* rec_pic_y = rec_pic->getAddr(CompID);
	//int height = rec_pic->getHeight(CompID);
	int width = rec_pic->getWidth(CompID);
	int strideY = rec_pic->getStride(CompID);
	//int stride_dst = patch.endX - patch.startX + 16;
	src += 16 - patch.startX + (16 - patch.startY) * stride_dst;

	TComPicYuv* pred_pic = pcPic->p_pcPicYuvPred;
	Pel* pred_pic_y = pred_pic->getAddr(CompID);
	int strideP = pred_pic->getStride(CompID);

	static int six = 2 + ::log2(pcPic->getCtu(0)->getTotalNumPart()) / 2;
	int ctus_in_width = ((width - 1) >> six) + 1;

	int pLineRec, pLineDst;
	if (pcPic->getSlice(0)->getSliceType() == I_SLICE)
	{
		for (int i = patch.startY; i < patch.endY; i++)
			for (int j = patch.startX; j < patch.endX; j++)
				rec_pic_y[i*strideY + j] = Clip3((Pel)0, (Pel)255, (Pel)src[i*stride_dst + j]);
	}
	else
	{
		int ctu_idx, row, column, z;
		for (int i = patch.startY; i < patch.endY; i++)
		{
			int line_ctu_idx = (i >> six)* ctus_in_width;
			row = (i & ((1 << six) - 1)) >> 2;
			pLineRec = i * strideY;
			pLineDst = i * stride_dst;
			for (int j = patch.startX; j < patch.endX; j++)
			{
				ctu_idx = line_ctu_idx + (j >> six);
				column = (j & ((1 << six) - 1)) >> 2;
				z = pcPic->m_RasterToZscan[row*(1 << (six - 2)) + column];
				auto ctu = pcPic->getCtu(ctu_idx);
				if //(ctu->isIntra(z) || (ctu->getCbf(z, CompID, ctu->getTransformIdx(z))))
				     (pred_pic_y[i*strideP+j] != rec_pic_y[pLineRec + j])
					rec_pic_y[pLineRec + j] = Clip3((Pel)0, (Pel)255, (Pel)src[pLineDst + j]);
			}
		}
	}
}

_declspec(dllexport) void run_model(TComPic* pcPic, TComPicYuv* dstPic)
{
	TComPicYuv* rec_pic = pcPic->getPicYuvRec();
	
	static char* files[] = { 
		"Parameters_ai_qp22.pa", //0
		"Parameters_ai_qp28.pa", //1
		"Parameters_ai_qp32.pa", //2
		"Parameters_ai_qp36.pa", //3
		"Parameters_ai_qp40.pa", //4
		"Parameters_ai_qp45.pa"  //5
	};
	static int previous_fidx = 0;
	int fidx = 0;
	//if (pcPic->getSlice(0)->getSliceType() == I_SLICE)
	{
		if (pcPic->getSlice(0)->getSliceQp() < 27)
			fidx = 0;
		else
		if (pcPic->getSlice(0)->getSliceQp() < 32)
			fidx = 1;
	    else if (pcPic->getSlice(0)->getSliceQp() < 36)
			fidx = 2;
		else if (pcPic->getSlice(0)->getSliceQp() < 40)
			fidx = 3;
		else
			fidx = 4;
	}
	/*/else if (pcPic->getSlice(0)->getSliceType() == P_SLICE)
	{
		if (pcPic->getSlice(0)->getSliceQp() < 32)
			fidx = 0;
		else if (pcPic->getSlice(0)->getSliceQp() < 36)
			fidx = 1;
		else if (pcPic->getSlice(0)->getSliceQp() < 41)
			fidx = 2;
		else
			fidx = 3;
	}/*/

	bool use_gpu = torch::cuda::is_available();
	static Module* ppf[5] = { NULL, NULL, NULL, NULL, NULL};
	auto& pf = ppf[fidx];
	cout << files[fidx] << endl;
	if ((pf == NULL)) //|| (fidx != previous_fidx))
	{
		cout << "Creating Module." << endl;
		previous_fidx = fidx;
		FILE* fid = NULL;
		fid = fopen(files[fidx], "rb");
		if (fid == NULL)
		{
			std::cout << std::endl << "Error: failed to open parameters file!" << std::endl;
			return;
		}
		if (pf != NULL)
		{
			if (use_gpu)
			{
				delete pf;
				pf = new Module;
			}
		}
		else
		{
			pf = new Module;
		}
		auto& f = *pf;

		for (auto param : f.parameters())
		{
			param.set_requires_grad(false);
			// param.zero_();
			fread(param.data_ptr(), sizeof(float), param.numel(), fid);
			//::print(param);
		}
		fclose(fid);
		if (use_gpu) {
			f.to(at::kCUDA);
			f.gpu = true;
		}
	}
	auto& f = *pf;
	
	int picHeight = rec_pic->getHeight(COMPONENT_Y);
	int picWidth = rec_pic->getWidth(COMPONENT_Y);

	Rect patch = {0,0,0,0};
	int width, height;
	//if (picWidth < 352)
	//	width = picWidth + 42;
	//else
	//	width = 352 + 42;
	//if (picHeight < 288)
	//	height = picHeight + 42;
	//else 
	//	height = 288 + 42;
	const int PSW = 128; // 448;
	const int PSH = 128; // 256;

	for (patch.startX = 0; patch.startX < picWidth; patch.startX += PSW)
	{
		if (patch.startX + PSW < picWidth )//- 32)
			patch.endX = patch.startX + PSW;
		else 
		{
			patch.endX = picWidth;
			patch.startX = picWidth - PSW; //
			picWidth = 0;
		}
        width = patch.endX - patch.startX + 42;
		picHeight = rec_pic->getHeight(COMPONENT_Y);
		for (patch.startY = 0; patch.startY < picHeight; patch.startY += PSH)
		{
			if (patch.startY + PSH < picHeight )//- 32)
				patch.endY = patch.startY + PSH;
			else
			{
				patch.endY = picHeight;
				patch.startY = picHeight - PSH; //
				picHeight = 0;
			}
			height = patch.endY - patch.startY + 42;

			torch::Tensor x = torch::zeros({ 1, 2, height, width });
			pic_cpy((float_t*)x[0][0].data_ptr(), rec_pic, patch, width);
			pic_cpy((float_t*)x[0][1].data_ptr(), pcPic->p_pcPicYuvPred, patch, width);
			//static torch::Tensor x = torch::zeros({ 1, 2, height, width }).to(at::kCUDA); 
			if (use_gpu)
			    x = x.to(at::kCUDA);
			    //at::CopyBytes(xc.numel() * sizeof(float_t), xc.data_ptr(), kCPU, x.data_ptr(), kCUDA, false);
			x /= 256;
			
			auto y = f.forward(x);
			y = (y * 256).round() ;

			if(use_gpu)
				y=y.to(at::kCPU);
			pic_cpy(pcPic, dstPic, (float_t*)y.data_ptr(), patch, width-10);
		}
	}
	if (fout.is_open()) fout.close();
}
