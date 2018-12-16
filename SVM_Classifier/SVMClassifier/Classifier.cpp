#include "Classifier.h"
#include <opencv2/highgui.hpp>
#include "../Helpers/ImageDataManager.h"
#include "../ImageProcessors/ImageProcessorBase/ImageProcessorBase.h"
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
	BowDictionary(bowDictionary);
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
			trainPaths.push_back(ImagePath{path, validImage.filename()});
		}
		return true;
	}

	return false;
}

void Classifier::SaveDictionary(Mat dictionary)
{
	FileStorage fs("dictionary.yml", FileStorage::WRITE);
	fs << "vocabulary" << dictionary;
	fs.release();
}

Mat Classifier::CreateBowDictionary()
{
	cout << "Building dictionary..." << endl;
	Mat features;
	for (auto trainPath : trainPaths)
	{
		cout << "Processing:\t" << trainPath.ImageFileName << endl;
		auto imageGroup = ImageDataManager::FetchImages(trainPath.DirectoryPath, trainPath.ImageFileName);
		SVMInput processedImage;
		imageProcessor->CalculateSVMInput(imageGroup, processedImage);
		features.push_back(processedImage.Data);
	}
	int dictionarySize = 200;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
	//retries number
	int retries = 1;
	//necessary flags
	int flags = KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flags);
	//cluster the feature vectors
	Mat bowDictionary = bowTrainer.cluster(features);
	//store the vocabulary
	SaveDictionary(bowDictionary);
	BowDictionary(bowDictionary);

	return Mat();
}

void Classifier::TrainSvm(Mat svmData, Mat_<float> svmLabels)
{
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::RBF;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_EPS, 100000 , 1e-9);
	params.gamma = 1; // for poly/rbf/sigmoid
	params.C = 1;
	

	//CvSVMParams params;
	//params.svm_type = CvSVM::C_SVC;
	//params.kernel_type = CvSVM::RBF;
	//params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-6);
	//params.gamma = 0.0001; // for poly/rbf/sigmoid



	cout << "Training svm" << endl;
	svm->train(svmData, svmLabels, Mat(), Mat(), params);

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
	TrainSvm(svmData, svmLabels);
	/*knn = CvKNearest();
	knn.train(svmData, svmLabels);*/

	try
	{
		cout << "Saving data for later :)";
		FileStorage fs("LBP_overlapp_data.xml", FileStorage::WRITE);
		fs << "LBP_overlapp_data" << svmData;
		fs.release();

		fs = FileStorage("LBP_overlapp_label.xml", FileStorage::WRITE);
		fs << "LBP_overlapp_label" << svmLabels;
		fs.release();
	}
	catch (...)
	{
	}
}

void Classifier::TrainBowSVM()
{
	Mat svmData;
	Mat_<float> svmLabels;
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create Sift feature point extracter
	Ptr<FeatureDetector> detector(new SiftFeatureDetector());
	//create Sift descriptor extractor
	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowImgDescriptorExtractor(extractor, matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowImgDescriptorExtractor.setVocabulary(bowDictionary);
	for (auto trainPath : trainPaths)
	{
		cout << "Calculating:\t" << trainPath.ImageFileName << endl;
		auto imageGroup = ImageDataManager::FetchImages(trainPath.DirectoryPath, trainPath.ImageFileName);

		auto maskKeyPoints = imageProcessor->SplitAndCalculateKeyPoints(imageGroup.Image, imageGroup.Mask);
		if (maskKeyPoints.size() == 0) continue;

		Mat_<float> maskLabel = (Mat_<float>(1, 1) << 1);
		for (vector<KeyPoint> maskKeyPoint : maskKeyPoints)
		{
			Mat bowDescriptor;
			//extract BoW (or BoF) descriptor from given image
			bowImgDescriptorExtractor.compute(imageGroup.Image, maskKeyPoint, bowDescriptor);
			svmData.push_back(bowDescriptor);

			svmLabels.push_back(maskLabel);
		}

		auto backgroundMaskKeyPoints = imageProcessor->SplitAndCalculateKeyPoints(
			imageGroup.Image, imageGroup.BackgroundMask, maskKeyPoints.size());

		if (backgroundMaskKeyPoints.size() == 0) continue;

		Mat_<float> backgroundMaskLabel = (Mat_<float>(1, 1) << 0);
		for (vector<KeyPoint> backgroundMaskKeyPoint : backgroundMaskKeyPoints)
		{
			Mat bowDescriptor;
			//extract BoW (or BoF) descriptor from given image
			bowImgDescriptorExtractor.compute(imageGroup.Image, backgroundMaskKeyPoint, bowDescriptor);
			svmData.push_back(bowDescriptor);
			svmLabels.push_back(backgroundMaskLabel);
		}

		/*auto maskDescriptors =imageProcessor->SplitAndCalculateKeyPoints(imageGroup.Image,imageGroup.Mask);
		if (maskDescriptors.size() == 0) continue;

		auto backgroundMaskDescriptors = imageProcessor->SplitAndCalculateKeyPoints(imageGroup.Image, imageGroup.BackgroundMask);
*/


		/*auto svmInput = imageProcessor->CalculateSVMInput(imageGroup);
		svmData.push_back(svmInput.Data);
		svmLabels.push_back(svmInput.Labels);*/
	}

	try
	{
		cout << "Saving data for later :)";
		cout << "Saving data for later :)";
		FileStorage fs("svmData_80pr_scale_20.xml", FileStorage::WRITE);
		fs << "svmData_80_scale_20" << svmData;
		fs.release();

		fs = FileStorage("svmLabel_80pr_scale_20.xml", FileStorage::WRITE);
		fs << "svmLabel_80_scale_20" << svmLabels;
		fs.release();
	}
	catch (...)
	{
	}


	


	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 500, 1e-6);

	cout << "Training svm" << endl;
	svm->train(svmData, svmLabels, Mat(), Mat(), params);
	cout << "Saving svm";
	svm->save("BOWSVM_80%_scale20.xml");
}

vector<float> Classifier::TestSVM(const path& testImage) const
{
	vector<float> results;
	auto image = imread(testImage.string(), CV_LOAD_IMAGE_COLOR);
	Mat processedImage;
	imageProcessor->TestImage(image, processedImage);
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

//vector<float> Classifier::TestBOWSVM(const path& testImage) const
//{
//	cout << "Testing SVM";
//	auto image = imread(testImage.string());
//	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
//	//create Sift feature point extracter
//	Ptr<FeatureDetector> detector(new SiftFeatureDetector());
//	//create Sift descriptor extractor
//	Ptr<DescriptorExtractor> extractor(new SiftDescriptorExtractor);
//	//create BoF (or BoW) descriptor extractor
//	BOWImgDescriptorExtractor bowImgDescriptorExtractor(extractor, matcher);
//	//Set the dictionary with the vocabulary we created in the first step
//	bowImgDescriptorExtractor.setVocabulary(bowDictionary);
//
//	Mat mask = Mat::ones(1080, 1920, CV_8UC1);
//	auto maskKeyPoints = imageProcessor->SplitAndCalculateKeyPoints(image, mask);
//
//	vector<float> predictions;
//	for (vector<KeyPoint> maskKeyPoint : maskKeyPoints)
//	{
//		Mat bowDescriptor;
//		//extract BoW (or BoF) descriptor from given image
//		bowImgDescriptorExtractor.compute(image, maskKeyPoint, bowDescriptor);
//		predictions.push_back(svm->predict(bowDescriptor));
//	}
//
//	/*int regionWidth = 1920 / 20;
//	int regionHeight = 1080 / 20;
//
//	Mat testMat = Mat::zeros(1080, 1920, CV_8UC1);
//	int i = 0;
//	Scalar scalar;
//	for (int y = 0; y < image.rows; y += regionHeight)
//	{
//	for (int x = 0; x < image.cols; x += regionWidth)
//	{
//	Rect rect = Rect(x, y, regionWidth, regionHeight);
//
//	if (predictions[i++] == 1)
//	scalar = Scalar(255, 255, 255);
//	else
//	scalar = Scalar(0, 0, 0);
//
//	rectangle(testMat, rect, scalar, -1);
//
//	}
//	}
//
//	imshow("!!!", testMat);
//	imwrite("testBOW.jpg", testMat);
//	waitKey();*/
//	return predictions;
//}

void Classifier::VisualizeClassification(const vector<float>& results, Mat &outputImage) const
{
	Mat resultImage;
	imageProcessor->DrawResults(results,resultImage);
	outputImage = resultImage;
	
	//imshow("testResult", resultImage);
	//imwrite("knn_result1.jpg", resultImage);
}

//void Classifier::VisualizeBOWClassification(vector<float> predictions,Mat &outputImage) const
//{
//	auto imageWidth = imageProcessor->Mesh().cols;
//	auto imageHeight = imageProcessor->Mesh().rows;
//	int scale = imageProcessor->RegionScale();
//
//	int regionWidth = imageWidth / scale;
//	int regionHeight = imageHeight / scale;
//
//	outputImage = Mat::zeros(1080, 1920, CV_8UC1);
//	int i = 0;
//	Scalar scalar;
//	for (int y = 0; y < imageHeight; y += regionHeight)
//	{
//		for (int x = 0; x < imageWidth; x += regionWidth)
//		{
//			cout << i;
//			Rect rect = Rect(x, y, regionWidth, regionHeight);
//			if (predictions[i++] == 1)
//				scalar = Scalar(255, 255, 255);
//			else
//				scalar = Scalar(0, 0, 0);
//
//			rectangle(outputImage, rect, scalar, -1);
//		}
//	}	
//}

void Classifier::LoadSVM(const path& svmPath)
{
	svm->load(svmPath.string().c_str());
}

void Classifier::LoadData(const Mat& data, const Mat& labels)
{

	TrainSvm(data, labels);

}

