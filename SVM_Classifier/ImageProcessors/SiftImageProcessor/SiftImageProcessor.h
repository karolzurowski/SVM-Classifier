#pragma once
#include <opencv2/core/core.hpp>
#include "../ImageProcessorBase/ImageProcessorBase.h"

class SiftImageProcessor:public ImageProcessorBase
{
public:
	SiftImageProcessor(int meshGap, int meshWidth, int meshHeight);
	void ProcessImage(const Mat& image,const Mat& mask, Mat& outputImage) override;
protected:
	void CalculateSIFT(const Mat& image, const Mat& mask, Mat& outputImage) const;
	void CalculateKeyPoints(const Mat& image, vector<KeyPoint>& keyPoints) const;
private:
	SiftFeatureDetector siftDetector;
};

