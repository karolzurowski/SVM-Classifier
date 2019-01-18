#pragma once

#include <opencv2/core/core.hpp>
#include "../SiftImageProcessor/SiftImageProcessor.h"

class MultiChannelImageProcessor:public SiftImageProcessor
{
public:
	MultiChannelImageProcessor(int meshGap, int imageWidth, int imageHeight);
	void ProcessImage(const Mat& image, const Mat& mask,Mat& outputImage) override;
};
