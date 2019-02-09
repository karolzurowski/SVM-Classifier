#pragma once
#include <opencv2/core/core.hpp>
#include "../ImageProcessorBase/ImageProcessorBase.h"

class SiftImageProcessor:public ImageProcessorBase
{
public:
	SiftImageProcessor( int meshWidth, int meshHeight,int meshGap );
	void ProcessImage(const Mat& image,const Mat& mask, Mat& outputImage) override;
protected:
	void CalculateSIFT(const Mat& image, const Mat& mask, Mat& outputImage) const;
	void CalculateKeyPoints(const Mat& image, vector<KeyPoint>& keyPoints) const;
private:
	SiftFeatureDetector siftDetector;
};

