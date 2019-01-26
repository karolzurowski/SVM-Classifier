#include "../../Helpers/HelperStructs.h"
#include <opencv2/contrib/contrib.hpp>
#include "BOWImageProcessor.h"
#include <iostream>
#include "../../Helpers/ImageDataManager.h"
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/highgui.hpp>
using namespace cv;


BOWImageProcessor::BOWImageProcessor(int meshGap, int imageWidth, int imageHeight) : SiftImageProcessor(
	meshGap, imageWidth, imageHeight), bowImgDescriptorExtractor(extractor, matcher)
{
//	bowImgDescriptorExtractor.setVocabulary(bowDictionary);
}

BOWImageProcessor::BOWImageProcessor(Mat& dictionary, int meshGap, int imageWidth, int imageHeight) : BOWImageProcessor(
	meshGap, imageWidth, imageHeight)
{
	BowDictionary(dictionary);
	bowImgDescriptorExtractor.setVocabulary(bowDictionary);
}

Mat BOWImageProcessor::CreateBowDictionary(const vector<ImagePath>& trainPaths,int dictionarySize)
{
	cout << "Building dictionary..." << endl;
	Mat features;
	for (auto trainPath : trainPaths)
	{
		cout << "Processing:\t" << trainPath.ImageFileName << endl;
		auto imageGroup = ImageDataManager::FetchImages(trainPath.DirectoryPath, trainPath.ImageFileName, CV_LOAD_IMAGE_COLOR);

		auto image = imageGroup.Image;
		auto mask = imageGroup.Mask;
		auto backgroundMask = imageGroup.BackgroundMask;

		Mat maskDescriptors;
		SiftImageProcessor::ProcessImage(image, mask, maskDescriptors);
		Mat backgroundMaskDescriptors;
		SiftImageProcessor::ProcessImage(image, backgroundMask, backgroundMaskDescriptors);

		Mat descriptors;
		descriptors.push_back(maskDescriptors);
		descriptors.push_back(backgroundMaskDescriptors);

		features.push_back(descriptors);
	}

	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_EPS, 1000, 1e-6);
	//retries number
	int retries = 1;
	//necessary flags
	int flags = KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	//cluster the feature vectors
	Mat bowDictionary = bowTrainer.cluster(features);
	//store the vocabulary
	BowDictionary(bowDictionary);
	bowImgDescriptorExtractor.setVocabulary(bowDictionary);
	SaveDictionary();
	
	return Mat();
}

void BOWImageProcessor::DrawResults(const vector<float>& results, Mat& outputImage)
{
	outputImage = Mat(1080, 1920, CV_8UC1);
	int regionWidth = imageWidth / regionScale;
	int regionHeight = imageHeight / regionScale;

	int resultIndex=0;

	for (int y = 0; y <= outputImage.rows - regionHeight; y += regionHeight)
	{
		for (int x = 0; x <= outputImage.cols - regionWidth; x += regionWidth)
		{
			Rect rect = Rect(x, y, regionWidth, regionHeight);
			if (results[resultIndex++] == 1)
			{
				rectangle(outputImage, rect, Scalar(255, 255, 255), -1);
			}
			else
			{
				rectangle(outputImage, rect, Scalar(0, 0, 0), -1);
			}
		}
	}

	imwrite("draw.jpg", outputImage);


}

void BOWImageProcessor::SaveDictionary()
{
	FileStorage fs("cylinder_dict.yml", FileStorage::WRITE);
	fs << "vocabulary" << bowDictionary;
	fs.release();
}

void BOWImageProcessor::ProcessImage(const Mat& image, const Mat& mask, Mat& outputImage)
{
	Mat svmData;
	Mat_<float> svmLabels;

	Mat grayScaleImage;
	cv::cvtColor(image, grayScaleImage, cv::COLOR_BGR2GRAY);

	//create BoF (or BoW) descriptor extractor

	//Set the dictionary with the vocabulary we created in the first step


	auto maskKeyPoints = SplitAndCalculateKeyPoints(grayScaleImage, mask);
	if (maskKeyPoints.size() == 0) return;

	Mat_<float> maskLabel = (Mat_<float>(1, 1) << 1);
	for (vector<KeyPoint> maskKeyPoint : maskKeyPoints)
	{
		Mat bowDescriptor;
		//extract BoW (or BoF) descriptor from given image
		bowImgDescriptorExtractor.compute(grayScaleImage, maskKeyPoint, bowDescriptor);
		outputImage.push_back(bowDescriptor);

		//svmLabels.push_back(maskLabel);
	}
}

vector<vector<KeyPoint>> BOWImageProcessor::SplitAndCalculateKeyPoints(const Mat& image, const Mat& mask, int vectorLimit) const
{
	vector < vector<KeyPoint>> keyPoints;
	int regionWidth = imageWidth / regionScale;
	int regionHeight = imageHeight / regionScale;

	float maxPoints = (float)regionWidth / meshGap * regionHeight / meshGap;
	int detectionThreshold = maxPoints * 0.9;

	vector<Mat> descriptors;


	for (int y = 0; y <= image.rows - regionHeight; y += regionHeight)
	{
		for (int x = 0; x <= image.cols - regionWidth; x += regionWidth)
		{
			Rect rect = Rect(x, y, regionWidth, regionHeight);
		


			Mat rectMask = Mat::zeros(1080, 1920, CV_8UC1);
			rectangle(rectMask, rect, Scalar(255, 255, 255), -1);

			Mat regionMeshed;
			multiply(rectMask, mask, regionMeshed);
			multiply(regionMeshed, mesh, regionMeshed);

			int objectPoints = countNonZero(regionMeshed);

			if (objectPoints < detectionThreshold) continue;

			//Mat processedRegion = ClassifyImage(image, rectMask);

			//descriptors.push_back(processedRegion);
			vector<KeyPoint> rectKeyPoints;
			CalculateKeyPoints(rectMask, rectKeyPoints);

			keyPoints.push_back(rectKeyPoints);
			if (vectorLimit > 0 && keyPoints.size() == vectorLimit)
				return keyPoints;
		}
	}
	return keyPoints;
	//	return descriptors;
}


