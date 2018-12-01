#include "ImageProcessor.h"
#include "../Helpers/HelperStructs.h"
#include <opencv2/contrib/contrib.hpp>
using namespace cv;


ImageProcessor::ImageProcessor(int meshGap, int meshWidth, int meshHeight) : ImageProcessorBase(
	meshGap, meshWidth, meshHeight)
{
}

Mat ImageProcessor::ProcessImage(const Mat& image, const Mat& mask)const
{	
	return CalculateSIFT(image, mask);
}







