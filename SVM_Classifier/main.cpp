#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "SVMClassifier/SVMClassifier.h"
#include "ImageProcessors/MultiChannelImageProcessor/MultiChannelImageProcessor.h"


using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
	try
	{
		/*	Mat dictionary;
			FileStorage fs("dictionary.yml", FileStorage::READ);
			fs["vocabulary"] >> dictionary;
			fs.release();*/
		SVMClassifier svm(make_unique<MultiChannelImageProcessor>(5, 1920, 1080));
		//svm.LoadSVM("DENSE_SIFT_badminton.xml");
		svm.AddTrainPath("C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\badminton_racket");
		//	svm.TrainBowSVM();
		//	svm.CreateBowDictionary();
		svm.TrainSvm();
		//	auto results = svm.TestSVM("C:\\Users\\karol\\OneDrive\\Dokumenty\\Visual Studio 2017\\Projects\\SVM_Classifier\\metal_cylinder0448.jpg");
			//svm.VisualizeClassification(results);
		auto results = svm.TestSVM("C:\\Users\\karol\\OneDrive\\Dokumenty\\Visual Studio 2017\\Projects\\SVM_Classifier\\badminton.jpg");
		svm.VisualizeClassification(results);


		//auto results=  svm.TestBOWSVM("C:\\Users\\karol\\OneDrive\\Dokumenty\\Visual Studio 2017\\Projects\\SVM_Classifier\\badminton.jpg");
		//svm.VisualizeClassification(results);
	//	svm.TrainBowSVM();

	






	}
	catch (filesystem::filesystem_error& error)
	{
		cout << error.what();
	}

	/*auto image = imread("C:\\Users\\karol\\OneDrive\\Dokumenty\\Visual Studio 2017\\Projects\\SVM_Classifier\\badminton.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	SiftFeatureDetector sift;
	vector<KeyPoint> keypoints;
	sift.detect(test, keypoints);
	drawKeypoints(test, keypoints, test);
	imshow("fasfs", test);*/


	waitKey(0);
	return 0;
}