#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "../Helpers/HelperStructs.h"
#include "ImageProcessorBase/ImageProcessorBase.h"

class ImageProcessor:public ImageProcessorBase
{
public:
	ImageProcessor(int meshGap, int meshWidth, int meshHeight);
	Mat ProcessImage(const Mat& image,const Mat& mask)const override;	
};

