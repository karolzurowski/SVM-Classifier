#include "ImageProcessor.h"
#include "../Helpers/HelperStructs.h"
#include <opencv2/contrib/contrib.hpp>
using namespace cv;


ImageProcessor::ImageProcessor(int meshGap, int meshWidth, int meshHeight) : ImageProcessorBase(
	meshGap, meshWidth, meshHeight)
{
}

void ImageProcessor::ProcessImage(const Mat& image, const Mat& mask, Mat& outputImage)const
{	
	Mat grayImg;
	cvtColor(image, grayImg,CV_BGR2GRAY);
	Mat thresholdedMask;
	threshold(mask, thresholdedMask, 100, 255, THRESH_BINARY);
	return CalculateSIFT(grayImg, thresholdedMask,outputImage);
}







