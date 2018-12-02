#pragma once
#include "../ImageProcessorBase/ImageProcessorBase.h"
#include <opencv2/core/core.hpp>

class MultiChannelImageProcessor:public ImageProcessorBase
{
public:
	MultiChannelImageProcessor(int meshGap, int meshWidth, int meshHeight);
	Mat ProcessImage(const Mat& image, const Mat& mask)const override;
};
