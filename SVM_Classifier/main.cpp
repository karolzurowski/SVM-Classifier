#include<iostream>
#include<opencv2\opencv.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "ImageDataManager.h"

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
	string inputPath = "C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\tire\\Masks";
	string outputPathString = "C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\tire\\Background_Masks";

	try
	{

		/*maskManager.CreateBackgroundMasks();*/
		//auto list = ImageDataManager::GetValidImageLists("C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\badminton_racket");
	//	cout << list[1];

		auto imageGroup = ImageDataManager::FetchImages(filesystem::path(
			R"(C:\Users\karol\OneDrive\Dokumenty\Dataset_Zakrzowek\Training_data\badminton_racket)"),
			"badminton_racket_0112.jpg");

		imshow("original", imageGroup.OriginalImage);
		imshow("mask", imageGroup.Mask);
		imshow("backgroundMask", imageGroup.BackgroundMask);	
		waitKey(0);

	}
	catch (filesystem::filesystem_error& error)
	{
		cout << error.what();
	}




	//	string sourceImgPath = "../Original_Images\\toilet_original.jpg";
	//	string maskPath = "../Test_Masks\\toilet_mask.jpg";
	//	string backgroundMaskPath = "../Output_Masks\\toilet_mask.jpg";
	//	string testImgPath = "../Original_Images\\toilet0319.jpg";
	//	Mat sourceImg = imread(sourceImgPath, CV_LOAD_IMAGE_COLOR);
	//	Mat maskImg = imread(maskPath, CV_LOAD_IMAGE_GRAYSCALE);
	//	Mat backgroundMaskImg = imread(backgroundMaskPath, CV_LOAD_IMAGE_GRAYSCALE);
	//	Mat testImg = imread(testImgPath, CV_LOAD_IMAGE_COLOR);
	//
	//	vector<Mat> channels(3);
	//
	//	split(sourceImg, channels);
	//
	//	sourceImg = channels[2];
	//	Mat sourceImg1 = channels[1];
	//	
	//	split(testImg, channels);
	//
	//	testImg = channels[2];
	//	Mat testImg1 = channels[1];
	//	/*cvtColor(sourceImg, sourceImg, CV_RGB2HSV);
	//	cvtColor(testImg, testImg, CV_RGB2GRAY);*/
	//
	//
	//
	//
	//
	//	int gridSize = 10;
	//	Mat grid = Mat::zeros(backgroundMaskImg.rows, backgroundMaskImg.cols, CV_8UC1);
	//
	//	for (size_t i = 0; i < backgroundMaskImg.rows; i++)
	//	{
	//		if (i%gridSize == 0)
	//		{
	//			for (size_t j = 0; j < backgroundMaskImg.cols; j++)
	//			{
	//				if (j%gridSize == 0)
	//					grid.at<uchar>(i, j) = 255;
	//			}
	//		}
	//	}
	//
	//	Mat meshedMainMask;
	//	multiply(maskImg, grid, meshedMainMask);
	//	
	//	Mat meshedBackgroundMask;
	//	multiply(backgroundMaskImg, grid, meshedBackgroundMask);
	//
	//	
	//	SiftFeatureDetector detector;
	//	//vector<cv::KeyPoint> keypoints;
	//	vector<Point> maskPoints;
	//	findNonZero(meshedMainMask, maskPoints);
	////detector.detect(sourceImg, keypoints, backgroundMaskImg);
	//	vector<KeyPoint> maskKeyPoints;
	//	for (auto point : maskPoints)
	//	{
	//		maskKeyPoints.push_back(KeyPoint(point, 1));
	//	}
	//
	//	vector<Point> backgroundMaskPoints;
	//	findNonZero(meshedBackgroundMask, backgroundMaskPoints);
	//
	//	vector<KeyPoint> backgroundMaskKeyPoints;
	//	for (auto point : backgroundMaskPoints)
	//		backgroundMaskKeyPoints.push_back(KeyPoint(point, 1));
	//
	//
	//
	//
	//
	//	//int i = 0;
	//	/*for ( i;i<500;i++)
	//	{
	//		try
	//		{
	//			image.at<Vec3b>(Point(keyPoints[i].x, keyPoints[i].y)) = Vec3b(255,255,255);
	//		}
	//		catch (...)
	//		{
	//			cout << "i: " << i << "\tcols: " << keyPoints[i].x << "\trows: " << keyPoints[i].x;
	//		}
	//	}
	//
	//	imwrite("result.png", image);
	//	imshow("toiletmask", image);
	//	waitKey(3000);*/
	//
	//	Mat maskDescriptors;
	//	Mat maskDescriptors1;
	//	detector.compute(sourceImg, maskKeyPoints, maskDescriptors);
	//	detector.compute(sourceImg1, maskKeyPoints, maskDescriptors1);
	//	
	//
	//	Mat backgroundMaskDescriptors;
	//	Mat backgroundMaskDescriptors1;
	//
	//	detector.compute(sourceImg, backgroundMaskKeyPoints, backgroundMaskDescriptors);
	//	detector.compute(sourceImg1, backgroundMaskKeyPoints, backgroundMaskDescriptors1);
	//
	//	Mat merged;
	//	merged.push_back(maskDescriptors);
	//	merged.push_back(backgroundMaskDescriptors);
	//
	//
	//	Mat merged1;
	//	merged1.push_back(maskDescriptors1);
	//	merged1.push_back(backgroundMaskDescriptors1);
	//
	//	hconcat(merged, merged1, merged);
	//
	//
	//	vector<float> labels;
	//	for (int i = 0; i < maskDescriptors.rows; i++)
	//		labels.push_back(1);
	//	for (int i = 0; i < backgroundMaskDescriptors.rows; i++)
	//		labels.push_back(0);
	//
	//	Mat labelsMat(labels);
	//
	//	
	//	Mat testMat = Mat::zeros(sourceImg.rows, sourceImg.cols, CV_8UC1);
	//
	//	//Mat meshedSource;
	//
	////	sourceImg.copyTo(meshedSource, grid);
	//
	//	vector<Point> sourcePoints;
	//	findNonZero(grid, sourcePoints);
	//	//detector.detect(sourceImg, keypoints, backgroundMaskImg);
	//	vector<KeyPoint> sourceKeyPoints;
	//	for (auto point : sourcePoints)
	//	{
	//		sourceKeyPoints.push_back(KeyPoint(point, 1));
	//	}
	//
	//	Mat sourceDescriptors;
	//	Mat sourceDescriptors1;
	//
	//	detector.compute(testImg, sourceKeyPoints, sourceDescriptors);
	//	detector.compute(testImg1, sourceKeyPoints, sourceDescriptors1);
	//
	//	hconcat(sourceDescriptors, sourceDescriptors1, sourceDescriptors);
	//
	//	
	//	//imshow("mesheee", meshedSource);
	//	//waitKey(2000);
	//
	//	CvSVMParams params;
	//	params.svm_type = CvSVM::C_SVC;
	//	params.kernel_type = CvSVM::SIGMOID;
	//	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	//
	//	CvSVM SVM;
	//	SVM.train(merged, labelsMat, Mat(), Mat(), params);
	//
	//	vector<float> results;
	//	int c;
	//	for (int  i = 0; i < sourceDescriptors.rows; i++)
	//	{
	//		Mat row = sourceDescriptors.row(i);
	//		auto prediction = SVM.predict(row);
	//		/*if (prediction == 0)
	//			cout << i << "preditcion 0" << endl;
	//		else
	//			cout << i << "preditcion 1" << endl;*/
	//		results.push_back(SVM.predict(row));
	//	}
	//	
	//	
	//    for(int i=0;i<results.size();i++)
	//    {
	//		
	//		if (results[i] == 1)
	//			grid.at<uchar>(sourcePoints[i].y, sourcePoints[i].x) = 255;
	//		else
	//			grid.at<uchar>(sourcePoints[i].y, sourcePoints[i].x) = 0;
	//    }
	//
	//	imshow("!!!!!!!!!", grid);
	//	imwrite("result.jpg", grid);

		//cout << descriptors.type();

		//vector<vector<float>> vectorDescriptors;

		//for (int i = 0; i < descriptors.rows; i++)
		//{
		//	vector<float> descriptor;
		//	for (int j = 0; j < 128; j++)
		//	{
		//		descriptor.push_back( descriptors.at<float>(i, j));
		//	}
		//	vectorDescriptors.push_back(descriptor);
		//}




		//Mat output;
		//drawKeypoints(sourceImg, keypoints, output);



		/*vector<Point> keyPoints;
		findNonZero(backgroundMaskImg, keyPoints);


		namedWindow("win",CV_WINDOW_FREERATIO);
		imshow("win", output);
		imwrite("outputKeypoints.jpg", output);
		imshow("original", sourceImg);
		imshow("mask", maskImg);
		imshow("backgroundMask", backgroundMaskImg);*/


	waitKey(0);
	return 0;
}