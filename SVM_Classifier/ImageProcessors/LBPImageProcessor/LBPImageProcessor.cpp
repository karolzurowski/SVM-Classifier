#include "LBPImageProcessor.h"
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <opencv2/highgui.hpp>

LBPImageProcessor::LBPImageProcessor( int meshWidth, int meshHeight, int meshGap) : ImageProcessorBase(
	meshWidth, meshHeight, meshGap)
{
	//regionScale = 20;
}

void LBPImageProcessor::ProcessImage(const Mat& image, const Mat& mask, Mat& outputImage)
{

	//Mat_<float> histograms;
	outputImage.convertTo(outputImage, CV_32FC1);
	//Mat lbpImage1;
	/*CalculateLBP(image, lbpImage1);
	imwrite("LBP_color.jpg", lbpImage1);*/

	//vector<Mat> channels;
	//split(image, channels);
	//Mat lbp1;
	//Mat lbp2;
	//Mat lbp3;
	//CalculateLBP(channels[0], lbp1);
	//CalculateLBP(channels[1], lbp2);
	//CalculateLBP(channels[2], lbp3);

	//imwrite("lbp1.jpg", lbp1);
	//imwrite("lbp2.jpg", lbp2);
	//imwrite("lbp3.jpg", lbp3);


	Mat grayImage;
	cvtColor(image, grayImage, CV_BGR2GRAY);

	Mat thresholdedMask;
	threshold(mask, thresholdedMask, 200, 255, THRESH_BINARY);


	/*Mat lbpImage;
	CalculateLBP(grayImage, lbpImage);
	imwrite("LBP_gray.jpg", lbpImage);*/
	vector < vector<KeyPoint>> keyPoints;
	int regionWidth = imageWidth / regionScale;
	int regionHeight = imageHeight / regionScale;

	float maxPoints = (float)regionWidth / meshGap * regionHeight / meshGap;
	int detectionThreshold = maxPoints * 0.8;

	vector<Mat> descriptors;
	for (int y = 0; y <= image.rows - regionHeight; y += 54)
	{
		for (int x = 0; x <= image.cols - regionWidth; x += 96)
		{
			/*for (int y = 0; y < grayImage.rows - regionHeight; y +=  +50)
			{
				for (int x = 0; x < grayImage.cols- regionWidth; x += 50)
				{*/
			Rect rect = Rect(x, y, regionWidth, regionHeight);
			Mat rectMat(grayImage, rect);

			Mat rectMask = Mat::zeros(1080, 1920, CV_8UC1);
			rectangle(rectMask, rect, Scalar(255, 255, 255), -1);


			multiply(rectMask, thresholdedMask, rectMask);

			int objectPoints = countNonZero(rectMask);

			if (objectPoints < detectionThreshold)
			{
				;
				continue;
			}
			//Mat processedRegion = ClassifyImage(image, rectMask);
			Mat lbpImage;
			//descriptors.push_back(processedRegion);
			CalculateLBP(rectMat, lbpImage);
			Mat histogram;
			CalculateHistogram(lbpImage, histogram, 256);
			//	cout << histogram << endl;
			//histograms.push_back(histogram);
			outputImage.push_back(histogram);
		}

	}
}

void LBPImageProcessor::DrawResults(const vector<float>& results, Mat& mat)
{
	int regionWidth = imageWidth / regionScale;
	int regionHeight = imageHeight / regionScale;
	mat = Mat::zeros(1080, 1920, CV_8UC1);
	Scalar scalar;
	int index = 0;
	for (int y = 0; y <= mat.rows - regionHeight; y += 54)
	{
		for (int x = 0; x <= mat.cols - regionWidth; x += 96)
		{
			Rect rect = Rect(x, y, regionWidth, regionHeight);
			if (results[index++] == 1)
				scalar = Scalar(255, 255, 255);
			else
				scalar = Scalar(0, 0, 0);

			rectangle(mat, rect, scalar, -1);
		}
	}

}

//void LBPImageProcessor::ClassifyImage(const Mat& image, Mat & outputImage)
//{
//	cout << "testing image..." << endl;
//	outputImage.convertTo(outputImage, CV_8UC1);
//	//Mat_<float> histograms;
//
//
//
//	Mat defaultMask = Mat::zeros(1080, 1920, CV_8UC1);
//	defaultMask.setTo(cv::Scalar(255, 255, 255));
//
//	Mat grayImage;
//	cvtColor(image, grayImage, CV_BGR2GRAY);
//
//	/*Mat lbpImage;
//	CalculateLBP(grayImage, lbpImage);
//	imwrite("LBP_gray.jpg", lbpImage);*/
//	vector < vector<KeyPoint>> keyPoints;
//	int regionWidth = imageWidth / regionScale;
//	int regionHeight = imageHeight / regionScale;
//
//
//
//	vector<Mat> descriptors;
//	//for (int y = 0; y < (image.rows - regionHeight / 2); y += regionHeight / 2)
//	//{
//	//	for (int x = 0; x < (image.cols - regionWidth / 2); x += regionWidth / 2)
//	//	{
//	for (int y = 0; y < (image.rows - regionHeight / 2); y += regionHeight / 2)
//	{
//		for (int x = 0; x < (image.cols - regionWidth / 2); x += regionWidth / 2)
//		{
//			Rect rect = Rect(x, y, regionWidth, regionHeight);
//			Mat rectMat(grayImage, rect);
//
//			Mat rectMask = Mat::zeros(1080, 1920, CV_8UC1);
//			rectangle(rectMask, rect, Scalar(255, 255, 255), -1);
//
//
//			multiply(rectMask, defaultMask, rectMask);
//
//
//
//			//	!!!!!!!!!!!!!!!!	//	if (objectPoints < detectionThreshold) continue;
//
//
//						//Mat processedRegion = ClassifyImage(image, rectMask);
//			Mat lbpImage;
//			//descriptors.push_back(processedRegion);
//			CalculateLBP(rectMat, lbpImage);
//			Mat histogram;
//			CalculateHistogram(lbpImage, histogram, 256);
//			//	cout << histogram << endl;
//			outputImage.push_back(histogram);
//
//		}
//	}
//
//
//}



void LBPImageProcessor::CalculateHistogram(const Mat& src, Mat& hist, int numPatterns) const
{
	hist = Mat::zeros(1, numPatterns, CV_8UC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int bin = src.at<uchar>(i, j);
			hist.at<uchar>(0, bin) += 1;
		}
	}
}


void LBPImageProcessor::CalculateLBP(const Mat& inputImage, Mat& outputImage) const {
	outputImage = Mat::zeros(inputImage.rows - 2, inputImage.cols - 2, CV_8UC1);
	for (int i = 1; i < inputImage.rows - 1; i++) {
		for (int j = 1; j < inputImage.cols - 1; j++) {
			uchar center = inputImage.at<uchar>(i, j);
			unsigned char code = 0;
			code |= (inputImage.at<uchar>(i - 1, j - 1) > center) << 7;
			code |= (inputImage.at<uchar>(i - 1, j) > center) << 6;
			code |= (inputImage.at<uchar>(i - 1, j + 1) > center) << 5;
			code |= (inputImage.at<uchar>(i, j + 1) > center) << 4;
			code |= (inputImage.at<uchar>(i + 1, j + 1) > center) << 3;
			code |= (inputImage.at<uchar>(i + 1, j) > center) << 2;
			code |= (inputImage.at<uchar>(i + 1, j - 1) > center) << 1;
			code |= (inputImage.at<uchar>(i, j - 1) > center) << 0;
			outputImage.at<unsigned char>(i - 1, j - 1) = code;
		}
	}
}