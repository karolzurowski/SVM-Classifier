#include<iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ImageProcessors/ImageProcessor.h"
#include "SVMClassifier/SVMClassifier.h"

using namespace cv;
using namespace std;


int main(int argc, char **argv)
{
	try
	{		
		SVMClassifier svm(make_unique<ImageProcessor>(5, 1920, 1080));
		svm.AddTrainPath("C:\\Users\\karol\\OneDrive\\Dokumenty\\Visual Studio 2017\\Projects\\SVM_Classifier\\TEST_badminton");
		svm.TrainSvm();
		auto results=  svm.TestSVM("C:\\Users\\karol\\OneDrive\\Dokumenty\\Visual Studio 2017\\Projects\\SVM_Classifier\\badminton.jpg");
		svm.VisualizeClassification(results);

		waitKey(0);

	}
	catch (filesystem::filesystem_error& error)
	{
		cout << error.what();
	}





	waitKey(0);
	return 0;
}