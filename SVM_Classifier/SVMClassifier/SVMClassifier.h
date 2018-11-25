#pragma once
#include "../Helpers/HelperStructs.h"
#include <opencv2/core/affine.hpp>
#include <filesystem>
#include <opencv2/ml/ml.hpp>
using namespace std;
using namespace filesystem;

class ImageProcessorBase;

class SVMClassifier
{
public:
	SVMClassifier(std::unique_ptr<ImageProcessorBase>&& _imageProcessor);
	bool AddTrainPath(const path& imagesDirectory);
	void VisualizeClassification(const vector<float>& results) const;
	vector<float> TestSVM(const path& testImage) const;
	void TrainSvm();

private:
	unique_ptr<ImageProcessorBase> imageProcessor;
	unique_ptr<CvSVM> svm;
	vector<ImagePath> trainPaths;
};
