#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "SVMClassifier/Classifier.h"
#include "ImageProcessors/SiftImageProcessor/SiftImageProcessor.h"
#include "ImageProcessors/LBPImageProcessor/LBPImageProcessor.h"
#include "ImageProcessors/MultiChannelImageProcessor/MultiChannelImageProcessor.h"
#include "ImageProcessors/BOWImageProcessor/BOWImageProcessor.h"

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
	try
	{
			Mat dictionary;
			FileStorage fs("cylinder_dict.yml", FileStorage::READ);
			fs["vocabulary"] >> dictionary;
			fs.release();

			//Mat data;
			//FileStorage fs("LBP_overlapp_data.xml", FileStorage::READ);
			//fs["LBP_overlapp_data"] >> data;
			//fs.release();

			//Mat label;
			//FileStorage fs1("LBP_overlapp_label.xml", FileStorage::READ);
			//fs1["LBP_overlapp_label"] >> label;
			//fs1.release();


	//	Classifier classifier(make_unique<BOWImageProcessor>(dictionary, 1920, 1080,10));
		Classifier classifier(make_unique<BOWImageProcessor>( 1920, 1080,10));
		classifier.LoadSVM("SVM_SIFT.xml");

		//lassifier.LoadData(data, label);

		classifier.AddTrainPath("C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\metal_cylinder_2");
		classifier.AddTestPath("C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\small\\cylinder");
		//	svm.TrainBowSVM();
			classifier.CreateBowDictionary();

		//classifier.CreateBowDictionary();
	classifier.TrainSvm();
	//	classifier.TestImages();

	//	string testImagePath = "C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\small\\one_racket\\Original_Frames\\badminton_racket_0149.jpg";
		//	//string testImagePath1 = "C:\\Users\\karol\\Desktop\\metal_cylinder0306.jpg";
		//	//auto results = classifier.TestImage(testImagePath);
		//Mat testMat;
		////	classifier.VisualizeBOWClassification(results,testMat);
		////	classifier.VisualizeClassification(results,testMat);
		//	namedWindow("win", CV_WINDOW_NORMAL);
		//	imshow("win", testMat);
		//	imwrite("SIFT_cylinder.jpg", testMat);
		//	waitKey();



		//auto results=  svm.TestBOWSVM("C:\\Users\\karol\\OneDrive\\Dokumenty\\Visual Studio 2017\\Projects\\SVM_Classifier\\badminton.jpg");
		//svm.VisualizeClassification(results);
		//	svm.TrainBowSVM();

	}
	catch (filesystem::filesystem_error& error)
	{
		cout << error.what();
	}

	waitKey(0);
	return 0;
}