#pragma once
#include "../Helpers/HelperStructs.h"
#include <opencv2/core/affine.hpp>
#include <filesystem>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace filesystem;

class ImageProcessorBase;

class Classifier
{
public:
	Classifier(unique_ptr<ImageProcessorBase> _imageProcessor);
	Classifier(unique_ptr<ImageProcessorBase> _imageProcessor, Mat bowDictionary);
	Classifier(unique_ptr<ImageProcessorBase> _imageProcessor, Mat bowDictionary, string svmPath);
	bool AddTrainPath(const path& imagesDirectory);
	bool AddTestPath(const path& imagesDirectory);
	void TestImages();
	
	void CreateBowDictionary();
	void TrainSvm(Mat svmData, Mat_<float> svmLabels);
	void VisualizeClassification(const vector<float>& results, Mat &outputImage) const;
	vector<float> TestImage(const Mat& testImage) const;
	//void VisualizeBOWClassification(vector<float> predictions,Mat &outputImage) const;
	//vector<float> TestBOWSVM(const path& testImage) const;
	void TrainSvm();	
	
	void LoadSVM(const path& svmPath);
	

	void LoadData(const Mat& data, const Mat& label);


private:
	unique_ptr<ImageProcessorBase> imageProcessor;
	unique_ptr<CvSVM> svm;
	//int patchScale = 20;
	vector<ImagePath> trainPaths;
	vector<ImagePath> testPaths;
	
	
	KNearest knn;
};
