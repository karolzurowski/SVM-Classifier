#pragma once
#pragma once
#include "../ImageProcessorBase/ImageProcessorBase.h"
#include <opencv2/core/core.hpp>

class LBPImageProcessor :public ImageProcessorBase
{
public:
	LBPImageProcessor(int meshGap, int meshWidth, int meshHeight);
	void ProcessImage(const Mat& image, const Mat& mask, Mat& outputImage) override;
	void DrawResults(const vector<float>& results, Mat& mat) override;

	//void ClassifyImage(const Mat& image,Mat& outputImage)  override;
protected:
	void CalculateLBP(const Mat& inputImage, Mat& outputImage) const;
	Mat Mesh() const override { return Mat(); }
	
protected:
	void CalculateHistogram(const Mat& src, Mat& hist, int numPatterns) const;
};
