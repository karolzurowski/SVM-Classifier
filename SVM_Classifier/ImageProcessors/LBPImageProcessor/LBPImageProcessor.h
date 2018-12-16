#pragma once
#pragma once
#include "../ImageProcessorBase/ImageProcessorBase.h"
#include <opencv2/core/core.hpp>

class LBPImageProcessor :public ImageProcessorBase
{
public:
	LBPImageProcessor(int meshGap, int meshWidth, int meshHeight);
	void ProcessImage(const Mat& image, const Mat& mask, Mat& outputImage)const override;
	void DrawResults(const vector<float>& results, Mat& mat) override;

	void TestImage(const Mat& image,Mat& outputImage) const override;
protected:
	void CalculateLBP(const Mat& inputImage, Mat& outputImage) const;

	
protected:
	void CalculateHistogram(const Mat& src, Mat& hist, int numPatterns) const;
};
