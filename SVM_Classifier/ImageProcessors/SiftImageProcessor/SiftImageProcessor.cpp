#include "../../Helpers/HelperStructs.h"
#include <opencv2/contrib/contrib.hpp>
#include "SiftImageProcessor.h"
#include <opencv2/highgui.hpp>
using namespace cv;


SiftImageProcessor::SiftImageProcessor( int meshWidth, int meshHeight, int meshGap) : ImageProcessorBase(
	meshWidth, meshHeight,meshGap)
{
}

void SiftImageProcessor::ProcessImage(const Mat& image, const Mat& mask, Mat& outputImage)
{	
	Mat grayImg;
	cvtColor(image, grayImg,CV_BGR2GRAY);
	Mat thresholdedMask;
	threshold(mask, thresholdedMask, 200, 255, THRESH_BINARY);
	return CalculateSIFT(grayImg, thresholdedMask,outputImage);
}

void SiftImageProcessor::CalculateSIFT(const Mat& image, const Mat& mask, Mat& outputImage) const
{
	vector<KeyPoint> keyPoints;
	CalculateKeyPoints(mask, keyPoints);
	//siftDetector.detect(image, keyPoints, mask);
	siftDetector.compute(image, keyPoints, outputImage);
}

void SiftImageProcessor::CalculateKeyPoints(const Mat& image, vector<KeyPoint>& keyPoints) const
{
	vector<Point> imagePoints;
	Mat meshedImage;
	multiply(image, mesh, meshedImage);
	findNonZero(meshedImage, imagePoints);
	for (auto point : imagePoints)
	{
		keyPoints.push_back(KeyPoint(point, meshGap));
	}
}






