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

		Classifier classifier(make_unique<BOWImageProcessor>(15, 1920, 1080));
		classifier.CreateBowDictionary(2000);
		classifier.AddTrainPath("C:\\Users\\karol\\OneDrive\\Dokumenty\\Dataset_Zakrzowek\\Training_data\\metal_cylinder_2");
		classifier.TrainSvm();	
	
	return 0;
}