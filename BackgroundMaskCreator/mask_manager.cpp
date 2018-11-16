#include "mask_manager.h"
#include <filesystem>

#include <iostream>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;



MaskManager::MaskManager(const string& inputDirectoryPath, const string& outputDirectoryPath) : outputDirectoryPath(outputDirectoryPath)
{
	if (filesystem::exists(inputDirectoryPath) && filesystem::is_directory(inputDirectoryPath))
	{
		for (const auto& imgPath : filesystem::directory_iterator(inputDirectoryPath))
			imgPaths.push_back(imgPath.path());
	}
	else
	{
		throw filesystem::filesystem_error("Path not valid",error_code());
	}
}

void MaskManager::SaveImage(const filesystem::path& imgPath, Mat sourceImg) const
{
	auto filename = imgPath.filename();
	auto outputPath = (outputDirectoryPath / filename);
	if (exists(outputPath))
	{
		cout << outputPath << "\t already exist! Saving aborted" << endl;
		return;
	}

	string outputString = outputPath.string();
	cout << "Saving:\t" << outputPath << endl;
	imwrite(outputString, sourceImg);
}

Mat MaskManager::CalculateBackgroundMask(Mat& inputImg, int minMaskSize, int maxMaskSize) const
{
	Mat tempImage;
	
	bitwise_not(inputImg, tempImage);
	distanceTransform(tempImage, tempImage, CV_DIST_L1, 3);	
	normalize(tempImage, tempImage, 0, 255, NORM_MINMAX);
	
	tempImage.convertTo(tempImage, CV_8UC1);
	
	int resultSize = 0;
	int distance = 500;
	int loops = 0;
	int threshold = 127;

	Mat outputImage;
	do
	{
		cv::threshold(tempImage, outputImage, threshold, 255, THRESH_BINARY);
		resultSize = countNonZero(outputImage);

		if (resultSize > maxMaskSize) threshold += 1;
		else threshold -= 1;

		loops++;

	}  //image is normalized from 0 to 255, starting value is 127 , so max loop number is 128
	while ((resultSize > maxMaskSize || resultSize < minMaskSize) && loops < 128);

	return outputImage;
}

void MaskManager::CreateBackgroundMask(const filesystem::path& imgPath) const
{
	string fullPath = imgPath.string();
	Mat sourceImg = imread(fullPath, CV_LOAD_IMAGE_GRAYSCALE);
	
	int imgSize = sourceImg.rows * sourceImg.cols;
	int maskSize = countNonZero(sourceImg);

	cv::threshold(sourceImg, sourceImg, initialThreshold, 255, THRESH_BINARY);	
	waitKey(3000);

	if (maskSize >= imgSize / 2)
	{
		
		cout << imgPath << "\t Mask size to big, inverting and saving..." << endl;
		bitwise_not(sourceImg, sourceImg);
		
		
		SaveImage(imgPath, sourceImg);
	}
	else
	{
		int maxBackgroundMaskSize = 1.1*maskSize;
		int minBackgroundMaskSize = 0.9*maskSize;

		Mat outputImg = CalculateBackgroundMask(sourceImg, minBackgroundMaskSize, maxBackgroundMaskSize);

		SaveImage(imgPath, outputImg);
	}
}

void MaskManager::CreateBackGroundMasks()
{
	Mat sourceImg;
	for (const auto & imgPath : imgPaths)
	{
		CreateBackgroundMask(imgPath);
	}
}
