#pragma once
#include <opencv2/core/core.hpp>
#include "ImageProcessorBase/ImageProcessorBase.h"

class ImageProcessor:public ImageProcessorBase
{
public:
	ImageProcessor(int meshGap, int meshWidth, int meshHeight);
	void ProcessImage(const Mat& image,const Mat& mask, Mat& outputImage)const override;	
};

