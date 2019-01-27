#include "Classifier.h"
#include <opencv2/highgui.hpp>
#include "../Helpers/ImageDataManager.h"
#include "../ImageProcessors/ImageProcessorBase/ImageProcessorBase.h"
#include <numeric>
#include <fstream>
#include "../ImageProcessors/BOWImageProcessor/BOWImageProcessor.h"
using namespace std;
using namespace filesystem;


Classifier::Classifier(unique_ptr<ImageProcessorBase> _imageProcessor)
	: imageProcessor(move(_imageProcessor))
{
	svm = make_unique<CvSVM>();
}

Classifier::Classifier(unique_ptr<ImageProcessorBase> _imageProcessor, Mat bowDictionary) : Classifier(
	move(_imageProcessor))
{

}

Classifier::Classifier(unique_ptr<ImageProcessorBase> _imageProcessor, Mat bowDictionary, string svmPath) : Classifier(
	move(_imageProcessor), bowDictionary)
{
	svm->load(svmPath.c_str());
}


bool Classifier::AddTrainPath(const path& path)
{
	auto validImages = ImageDataManager::GetValidImageLists(path);
	if (!validImages.empty())
	{
		for (auto validImage : validImages)
		{
			trainPaths.push_back(ImagePath{ path, validImage.filename() });
		}
		return true;
	}

	return false;
}

bool Classifier::AddTestPath(const path& path)
{
	auto validImages = ImageDataManager::GetValidImageLists(path);
	if (!validImages.empty())
	{
		for (auto validImage : validImages)
		{
			testPaths.push_back(ImagePath{ path, validImage.filename() });
		}
		return true;
	}

	return false;
}

void Classifier::TestImages()
{
	vector<float> precisions;
	vector<float> recalls;
	std::ofstream textFile("results.txt");

	for (auto testPath : testPaths)
	{
		cout << "testing:\t" << testPath.ImageFileName << endl;
		auto imageGroup = ImageDataManager::FetchImages(testPath.DirectoryPath, testPath.ImageFileName,
			CV_LOAD_IMAGE_COLOR);

		Mat mask;
		threshold(imageGroup.Mask, mask, 200, 255, THRESH_BINARY);
		auto results = TestImage(imageGroup.Image);
		Mat resultImage;
		imageProcessor->DrawResults(results, resultImage);

		Mat mesh = imageProcessor->Mesh();	
		Mat invertedMask;
		bitwise_not(imageGroup.Mask, invertedMask);
		Mat invertedResult;
		bitwise_not(resultImage, invertedResult);
		if (mesh.rows > 0)
		{
			multiply(imageGroup.Mask, mesh, mask);
			multiply(invertedMask, mesh, invertedMask);
			multiply(invertedResult, mesh, invertedResult);
		}


		Mat truePositivesImage;
		multiply(mask, resultImage, truePositivesImage);
		int truePositives = countNonZero(truePositivesImage);

		Mat falsePositivesImage;
		multiply(invertedMask, resultImage, falsePositivesImage);
		int falsePositives = countNonZero(falsePositivesImage);

		Mat trueNegativesImage;
		multiply(invertedMask, invertedResult, trueNegativesImage);
		int trueNegatives = countNonZero(trueNegativesImage);

		Mat falseNegativesImage;
		multiply(mask, invertedResult, falseNegativesImage);
		int falseNegatives = countNonZero(falseNegativesImage);

		int relevantElements = countNonZero(mask);
		int unrelevantElements = countNonZero(invertedMask);




		float precision = (float)truePositives / (truePositives + falsePositives);
		float recall = (float)truePositives / relevantElements;

		precisions.push_back(precision);
		recalls.push_back(recall);


		textFile << "True positive:\t" << truePositives << "\tFalse positives:\t" << falsePositives
			<< "\tTrue negatives:\t" << trueNegatives << "\tFalse negatives:\t" << falseNegatives 
			<<  "\tPrecision\t"<<precision<<"\tRecall:\t"<<recall<< endl;
		
		cout << "Precision\t" + std::to_string(precision) + "\tRecall:\t" + std::to_string(recall) << endl;
		
				imwrite("result.jpg", resultImage);
				imwrite("mask.jpg", imageGroup.Mask);
				imwrite("true_positive.jpg", truePositivesImage);
				imwrite("false_positive.jpg", falsePositivesImage);
				imwrite("false_negative.jpg", falseNegativesImage);
				imwrite("true_negative.jpg", trueNegativesImage);

	}

	float averagePrecision = std::accumulate(precisions.begin(), precisions.end(), 0.0) / precisions.size();
	float averageRecall = std::accumulate(recalls.begin(), recalls.end(), 0.0) / recalls.size();

	textFile << "Average precision: " << averagePrecision << endl;
	textFile << "Average recall: " << averageRecall << endl;
	textFile.close();


}

void Classifier::CreateBowDictionary(int dictionarySize)
{
	auto bowImageProcessor = dynamic_cast<BOWImageProcessor*>(imageProcessor.get());
	bowImageProcessor->CreateBowDictionary(trainPaths,dictionarySize);
}


void Classifier::TrainSvm(Mat svmData, Mat_<float> svmLabels)
{
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 1e-1);
	//params.gamma = 100; // for poly/rbf/sigmoid 
	//params.C =100; 
	//CvSVMParams params;
	//params.svm_type = CvSVM::C_SVC;
	//params.kernel_type = CvSVM::RBF;
	//params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-6);
	//params.gamma = 0.0001; // for poly/rbf/sigmoid

	cout << "Training svm" << endl;
	svm->train_auto(svmData, svmLabels, Mat(), Mat(), params, 2);

	//	svm->train_auto()
	//	knn = KNearest();
//knn.train(svmData, svmLabels);
	cout << "Saving svm";
	svm->save("SVM_LBP.xml");
}

void Classifier::TrainSvm()
{
	Mat svmData;
	Mat_<float> svmLabels;

	for (auto trainPath : trainPaths)
	{
		cout << "Calculating:\t" << trainPath.ImageFileName << endl;
		auto imageGroup = ImageDataManager::FetchImages(trainPath.DirectoryPath, trainPath.ImageFileName,
			CV_LOAD_IMAGE_COLOR);
		SVMInput svmInput;
		imageProcessor->CalculateSVMInput(imageGroup, svmInput);
		svmData.push_back(svmInput.Data);
		svmLabels.push_back(svmInput.Labels);
	}

	svmData.convertTo(svmData, CV_32FC1);
	svmLabels.convertTo(svmLabels, CV_32FC1);


	try
	{
		cout << "Saving data for later :)";
		FileStorage fs("SIFT_cylinder_data.xml", FileStorage::WRITE);
		fs << "SIFT_cylinder_data" << svmData;
		fs.release();

		fs = FileStorage("SIFT_cylinder_label.xml", FileStorage::WRITE);
		fs << "SIFT_cylinder_label" << svmLabels;
		fs.release();
	}
	catch (...)
	{
	}
	TrainSvm(svmData, svmLabels);
	/*knn = CvKNearest();
	knn.train(svmData, svmLabels);*/


}



vector<float> Classifier::TestImage(const Mat& testImage) const
{
	vector<float> results;

	Mat processedImage;
	imageProcessor->ClassifyImage(testImage, processedImage);
	processedImage.convertTo(processedImage, CV_32FC1);
	//vector<float> results;
	//for (int i = 0; i < processedImage.rows; i++)
	//{
	//	Mat row = processedImage.row(i);
	//	//	results.push_back(svm->predict(row));
	//	results.push_back(knn.find_nearest(row,5));
	//}

	//return results;

	//SiftFeatureDetector siftDetector;
	//Mat descriptors;
	//auto image = imread(testImage.string());
	////auto keyPoints = CalculateKeyPoints(mask);
	/*vector<KeyPoint> keyPoints;
	siftDetector.detect(image, keyPoints);
	siftDetector.compute(image, keyPoints, descriptors);*/
	//	Mat results = Mat::zeros(1080, 1920, CV_8UC1);
	for (int i = 0; i < processedImage.rows; i++)
	{
		Mat row = processedImage.row(i);
		//auto result = knn.find_nearest(row,5);
		auto result = svm->predict(row);
		results.push_back(result);


		/*if (result == 1)
			results.at<uchar>(keyPoints[i].pt.y, keyPoints[i].pt.x) = 255;
		else
			results.at<uchar>(keyPoints[i].pt.y, keyPoints[i].pt.x) = 0;		*/
	}



	return results;
}



void Classifier::VisualizeClassification(const vector<float>& results, Mat &outputImage) const
{
	Mat resultImage;
	imageProcessor->DrawResults(results, resultImage);
	outputImage = resultImage;

	//imshow("testResult", resultImage);
	//imwrite("knn_result1.jpg", resultImage);
}


void Classifier::LoadSVM(const path& svmPath)
{
	svm->load(svmPath.string().c_str());
}

void Classifier::LoadData(const Mat& data, const Mat& labels)
{

	TrainSvm(data, labels);

}

