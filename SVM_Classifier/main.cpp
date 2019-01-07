#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "SVMClassifier/Classifier.h"
#include "ImageProcessors/ImageProcessor.h"
#include "ImageProcessors/LBPImageProcessor/LBPImageProcessor.h"
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


		int a = 5;
		int b = 3;
		double result = (double)a / (a + b);


		//Mat data;
		//FileStorage fs("LBP_overlapp_data.xml", FileStorage::READ);
		//fs["LBP_overlapp_data"] >> data;
		//fs.release();

		//Mat label;
		//FileStorage fs1("LBP_overlapp_label.xml", FileStorage::READ);
		//fs1["LBP_overlapp_label"] >> label;
		//fs1.release();

	



		Classifier classifier(make_unique<ImageProcessor>(10, 1920, 1080));
		//classifier.LoadSVM("SVM_LBP.xml");

	//	classifier.LoadData(data, label);
//
		//classifier.AddTrainPath("C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\small\\one_cylinder");
		classifier.AddTrainPath("C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\small\\badminton_racket");
		classifier.AddTestPath("C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\small\\one_racket");
		//	svm.TrainBowSVM();
	//	classifier.CreateBowDictionary();
		classifier.TrainSvm();
		classifier.TestImages();
		
		string testImagePath = "C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\small\\one_racket\\Original_Frames\\badminton_racket_0149.jpg";
	//	//string testImagePath1 = "C:\\Users\\karol\\Desktop\\metal_cylinder0306.jpg";
	//	//auto results = classifier.TestImage(testImagePath);
 // 		Mat testMat;
	////	classifier.VisualizeBOWClassification(results,testMat);
	////	classifier.VisualizeClassification(results,testMat);
	//	namedWindow("win", CV_WINDOW_NORMAL);
	//	imshow("win", testMat);
	//	imwrite("SIFT_cylinder.jpg", testMat);
	//	waitKey();
		


			//auto results=  svm.TestBOWSVM("C:\\Users\\karol\\OneDrive\\Dokumenty\\Visual Studio 2017\\Projects\\SVM_Classifier\\badminton.jpg");
			//svm.VisualizeClassification(results);
		//	svm.TrainBowSVM();



			//Mat  testImage = imread("C:\\Users\\karol\\OneDrive\\Dokumenty\\Visual Studio 2017\\Projects\\SVM_Classifier\\badminton.jpg", CV_LOAD_IMAGE_GRAYSCALE);
			//Mat outputImage;
			//LBP(testImage, outputImage);
			//imshow("CalculateLBP", outputImage);
			//imwrite("CalculateLBP.jpg", outputImage);







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