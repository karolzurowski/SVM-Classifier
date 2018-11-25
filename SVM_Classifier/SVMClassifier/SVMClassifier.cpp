#include "SVMClassifier.h"
#include <opencv2/highgui.hpp>
#include "../Helpers/ImageDataManager.h"
#include "../ImageProcessors/ImageProcessorBase/ImageProcessorBase.h"
using namespace std;
using namespace filesystem;



SVMClassifier::SVMClassifier(std::unique_ptr<ImageProcessorBase>&& _imageProcessor)
	:imageProcessor(move(_imageProcessor))
{
	svm = make_unique<CvSVM>();
}


bool SVMClassifier::AddTrainPath(const path& path)
{
	auto validImages = ImageDataManager::GetValidImageLists(path);
	if (!validImages.empty())
	{
		for (auto validImage : validImages)
		{
			trainPaths.push_back(ImagePath{ path,validImage.filename() });
		}
		return true;
	}

	return false;
}

void SVMClassifier::VisualizeClassification(const vector<float>& results) const
{
	Mat grid(imageProcessor->Mesh());
	vector<Point> gridPoints;
	findNonZero(grid, gridPoints);

	for (int i = 0; i < results.size(); i++)
	{
		if (results[i] == 1)
			grid.at<uchar>(gridPoints[i].y, gridPoints[i].x) = 255;
		else
			grid.at<uchar>(gridPoints[i].y, gridPoints[i].x) = 0;
	}

	imshow("testResult", grid);
	//imwrite("SVMresult.jpg", grid);
}

vector<float>SVMClassifier::TestSVM(const path& testImage) const
{
	auto image = imread(testImage.string());
	SiftFeatureDetector siftDetector;
	auto processedImage = imageProcessor->ProcessImage(image);
	
	vector<float> results;
	for (int i = 0; i < processedImage.rows; i++)
	{
		Mat row = processedImage.row(i);		
		results.push_back(svm->predict(row));
	}

	return results;	
}

void SVMClassifier::TrainSvm()
{
	Mat svmData;
	Mat_<float> svmLabels;

	for (auto trainPath : trainPaths)
	{
		cout << "Calculating:\t" << trainPath.ImageFileName << endl;
		auto imageGroup = ImageDataManager::FetchImages(trainPath.DirectoryPath, trainPath.ImageFileName);
		auto svmInput = imageProcessor->CalculateSVMInput(imageGroup);
		svmData.push_back(svmInput.Data);
		svmLabels.push_back(svmInput.Labels);
	}

	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 500, 1e-6);

	cout << "Training svm" << endl;
	svm->train(svmData, svmLabels, Mat(), Mat(), params);

}
