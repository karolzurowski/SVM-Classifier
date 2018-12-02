#include "ImageProcessorBase.h"
#include <opencv2/highgui.hpp>
#include <iostream>

ImageProcessorBase::ImageProcessorBase(int meshGap, int meshWidth, int meshHeight) : imageWidth(meshWidth),
imageHeight(meshHeight)
{
	this->meshGap = meshGap;
	mesh = CreateMesh(meshGap, meshWidth, meshHeight);
}

Mat ImageProcessorBase::ProcessImage(const Mat& image) const
{
	return ProcessImage(image, mesh);
}

Mat ImageProcessorBase::CalculateSIFT(const Mat& image, const Mat& mask) const
{
	Mat descriptors;
	auto keyPoints = CalculateKeyPoints(mask);
	siftDetector.compute(image, keyPoints, descriptors);
	return descriptors;
}

Mat ImageProcessorBase::CalculateLabels(const Mat& maskDescriptors, const Mat& backgroundMaskDescriptors) const
{
	vector<float> labelsVector;
	for (int i = 0; i < maskDescriptors.rows; i++)
		labelsVector.push_back(1);
	for (int i = 0; i < backgroundMaskDescriptors.rows; i++)
		labelsVector.push_back(0);

	Mat labels(labelsVector);
	Mat labelsToReturn;
	labelsToReturn.push_back(labels);
	return labelsToReturn;
}

vector<vector<KeyPoint>> ImageProcessorBase::SplitAndCalculateKeyPoints(const Mat& image, const Mat& mask, int vectorLimit) const
{
	vector < vector<KeyPoint>> keyPoints;
	int regionWidth = imageWidth / regionScale;
	int regionHeight = imageHeight / regionScale;

	float maxPoints = (float)regionWidth / meshGap * regionHeight / meshGap;
	int detectionThreshold = maxPoints * 0.8;

	vector<Mat> descriptors;


	for (int y = 0; y < (image.rows - regionHeight / 2); y += regionHeight / 2)
	{
		for (int x = 0; x < (image.cols - regionWidth / 2); x += regionWidth / 2)
		{
			Rect rect = Rect(x, y, regionWidth, regionHeight);
			Mat rectMat(image, rect);


			Mat rectMask = Mat::zeros(1080, 1920, CV_8UC1);
			rectangle(rectMask, rect, Scalar(255, 255, 255), -1);

			Mat regionMeshed;
			multiply(rectMask, mask, regionMeshed);
			multiply(regionMeshed, mesh, regionMeshed);

			int objectPoints = countNonZero(regionMeshed);

			if (objectPoints < detectionThreshold) continue;

			//Mat processedRegion = ProcessImage(image, rectMask);

			//descriptors.push_back(processedRegion);
			keyPoints.push_back(CalculateKeyPoints(rectMask));
			if (vectorLimit > 0 && keyPoints.size() == vectorLimit)
				return keyPoints;
		}
	}
	return keyPoints;
	//	return descriptors;
}

vector<KeyPoint> ImageProcessorBase::CalculateKeyPoints(const Mat& image) const
{
	vector<KeyPoint> keyPoints;
	vector<Point> imagePoints;
	Mat meshedImage;
	multiply(image, mesh, meshedImage);
	findNonZero(meshedImage, imagePoints);
	for (auto point : imagePoints)
	{
		keyPoints.push_back(KeyPoint(point, meshGap));
	}
	return keyPoints;
}

SVMInput ImageProcessorBase::CalculateSVMInput(const ImageGroup& images) const
{
	//todo assert images not null

	auto image = images.Image;
	auto mask = images.Mask;
	auto backgroundMask = images.BackgroundMask;	
	Mat maskDescriptors = ProcessImage(image, mask);
	Mat backgroundMaskDescriptors = ProcessImage(image, backgroundMask);


	Mat resizedBackgroundDescriptors = Mat(backgroundMaskDescriptors);
	Mat resizedMaskDescriptors = Mat(maskDescriptors);
	if (backgroundMaskDescriptors.rows > maskDescriptors.rows)
	{
		resizedBackgroundDescriptors = Mat(maskDescriptors.rows, maskDescriptors.cols, backgroundMaskDescriptors.type(), backgroundMaskDescriptors.data);
		cout << "Changing background descriptors" << endl;
	//	cout << backgroundMaskDescriptors1.type();
	}
	else if( maskDescriptors.rows>backgroundMaskDescriptors.rows)
	{
		resizedMaskDescriptors = Mat(backgroundMaskDescriptors.rows, backgroundMaskDescriptors.cols, maskDescriptors.type(), maskDescriptors.data);
		cout << "Changing mask descriptors" << endl;
	}

	if(resizedBackgroundDescriptors.rows==  resizedMaskDescriptors.rows)
	{
		cout << "rows equal :)";
	}

	Mat descriptors;	
	descriptors.push_back(resizedMaskDescriptors);
	descriptors.push_back(resizedBackgroundDescriptors);

	Mat labelsToReturn = CalculateLabels(resizedMaskDescriptors, resizedBackgroundDescriptors);
	

	return SVMInput{ descriptors, labelsToReturn };
}

Mat ImageProcessorBase::CreateMesh(int meshGap, int meshWidth, int meshHeight) const
{
	Mat mesh = Mat::zeros(meshHeight, meshWidth, CV_8UC1);

	for (size_t i = 0; i < meshHeight; i++)
	{
		if (i % meshGap == 0)
		{
			for (size_t j = 0; j < meshWidth; j++)
			{
				if (j % meshGap == 0)
					mesh.at<uchar>(i, j) = 255;
			}
		}
	}
	return mesh;
}



/*   // get the image data
 int height = image.rows;
 int width = image.cols;

 printf("Processing a %dx%d image\n",height,width);

cv :: Size smallSize ( 110 , 70 );

std :: vector < Mat > smallImages ;
namedWindow("smallImages ", CV_WINDOW_AUTOSIZE );

for  ( int y =  0 ; y < image . rows ; y += smallSize . height )
{
	for  ( int x =  0 ; x < image . cols ; x += smallSize . width )
	{
		cv :: Rect rect =   cv :: Rect ( x , y , smallSize . width , smallSize . height );
		smallImages . push_back ( cv :: Mat ( image , rect ));
		imshow ( "smallImages", cv::Mat ( image, rect ));
		waitKey(0);
	}
}*/
