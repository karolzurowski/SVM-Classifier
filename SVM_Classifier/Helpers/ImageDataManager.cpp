#include <filesystem>

#include <iostream>
#include <opencv2/highgui.hpp>
#include "ImageDataManager.h"

using namespace cv;
using namespace std;
using namespace filesystem;


Mat ImageDataManager::CreateBackgroundMask(const path& imgPath)
{
	string fullPath = imgPath.string();
	Mat sourceImg = imread(fullPath, CV_LOAD_IMAGE_GRAYSCALE);

	if (sourceImg.data == NULL)
	{
		string error = imgPath.string() + "\tis invalid!";
		CV_Error(-1, "Invalid image file");
	}

	int imgSize = sourceImg.rows * sourceImg.cols;
	int maskSize = countNonZero(sourceImg);

	threshold(sourceImg, sourceImg, initialThreshold, 255, THRESH_BINARY);

	if (maskSize >= imgSize / 2)
	{
		cout << imgPath << "\t Mask size to big, inverting and saving..." << endl;
		bitwise_not(sourceImg, sourceImg);
		return sourceImg;
	}
	else
	{
		int maxBackgroundMaskSize = 1.1*maskSize;
		int minBackgroundMaskSize = 0.9*maskSize;
		return CalculateBackgroundMask(sourceImg, minBackgroundMaskSize, maxBackgroundMaskSize);
	}
}

Mat ImageDataManager::CalculateBackgroundMask(Mat& inputImg, int minMaskSize, int maxMaskSize)
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

vector<path> ImageDataManager::GetValidImageLists(path inputDirectoryPath, bool createBackgroundMasks)
{
	//todo find out why  "/" operator didnt work

	vector<path> validImages;
	if (exists(inputDirectoryPath) && is_directory(inputDirectoryPath))
	{
		auto directory = path(inputDirectoryPath);

		auto masksPath = path(directory).concat(masksDirectoryName);

		auto originalImagesPath = path(directory).concat(originalFramesDirectoryName);

		if (!exists(originalImagesPath) || !is_directory(originalImagesPath)) return validImages;
		if (!exists(masksPath) || !is_directory(masksPath)) return validImages;

		auto backgroundMasksPath = path(directory).concat(backgroundMasksDirectoryName);

		if (!exists(backgroundMasksPath) || !is_directory(backgroundMasksPath))
			if (!create_directory(backgroundMasksPath)) return validImages;

		for (const auto& image : directory_iterator(originalImagesPath))
		{
			if (!image.is_regular_file()) continue;

			auto filename = image.path().filename();
			auto maskFile = path(masksPath).concat("/" + filename.string());
			maskFile.replace_extension(".tif");
			if (!exists(maskFile) || !is_regular_file(maskFile)) continue;

			validImages.push_back(image);

			auto backgroundMaskFile = path(backgroundMasksPath).concat("/" + filename.string());
			backgroundMaskFile.replace_extension(".tif");

			if (!exists(backgroundMaskFile) || !is_regular_file(backgroundMaskFile))
			{

				if (createBackgroundMasks)
				{
					try
					{
						Mat resultImg = CreateBackgroundMask(maskFile);
						SaveImage(backgroundMaskFile, resultImg);
					}
					catch (const std::exception &ex)
					{
						cout << ex.what();
					}
				}
			}
		}
	}



	return validImages;
}

void ImageDataManager::SaveImage(const path& imagePath, Mat imageToSave)
{

	auto filename = imagePath.filename();
	if (exists(imagePath))
	{
		cout << imagePath << "\t already exist! Saving aborted" << endl;
		return;
	}

	string outputString = imagePath.string();
	cout << "Saving:\t" << imagePath << endl;

	imwrite(outputString, imageToSave);

}

ImageGroup ImageDataManager::FetchImages(const path directoryPath, const path imageName)
{
	ImageGroup fetchedImages;
	auto directory = directoryPath;
	auto filePath = "\\" + imageName.string();

	auto imageFile = directory;
	imageFile+= path(originalFramesDirectoryName) += path(filePath);
	if ((!exists(imageFile) || !is_regular_file(imageFile))) return fetchedImages;


	auto maskFile = directory;
	maskFile+= path(masksDirectoryName) +=path(filePath);
	maskFile.replace_extension(".tif");
	if ((!exists(maskFile) || !is_regular_file(maskFile)))  return fetchedImages;


	auto backgroundMaskFile = directory;
	backgroundMaskFile+= path(backgroundMasksDirectoryName) += path(filePath);
	backgroundMaskFile.replace_extension(".tif");
	if ((!exists(backgroundMaskFile) || !is_regular_file(backgroundMaskFile)))  return fetchedImages;

	fetchedImages.Image = imread(imageFile.string(), CV_LOAD_IMAGE_GRAYSCALE);
	fetchedImages.Mask = imread(maskFile.string(), CV_LOAD_IMAGE_GRAYSCALE);
	fetchedImages.BackgroundMask = imread(backgroundMaskFile.string(), CV_LOAD_IMAGE_GRAYSCALE);
	return fetchedImages;
}




