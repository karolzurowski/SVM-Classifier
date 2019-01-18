#include "ImageProcessorBase.h"
#include <opencv2/highgui.hpp>
#include <iostream>

ImageProcessorBase::ImageProcessorBase(int imageWidth, int ImageHeight, int meshGap) : imageWidth(imageWidth),
imageHeight(ImageHeight),meshGap(meshGap)
{
	if(meshGap>0) 	mesh = CreateMesh(meshGap, imageWidth, ImageHeight);
}

void ImageProcessorBase::ClassifyImage(const Mat& image, Mat& outputImage) 
{	
	ProcessImage(image, mesh, outputImage);
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



void ImageProcessorBase::DrawResults(const vector<float>& results,  Mat& resultImage)
{	
	resultImage = mesh.clone();
	vector<Point> gridPoints;
	findNonZero(resultImage, gridPoints);

	for (int i = 0; i < results.size(); i++)
	{
		if (results[i] == 1)
			resultImage.at<uchar>(gridPoints[i].y, gridPoints[i].x) = 255;
		else
			resultImage.at<uchar>(gridPoints[i].y, gridPoints[i].x) = 0;
	}
}




void ImageProcessorBase::CalculateSVMInput(const ImageGroup& images, SVMInput& svmInput) 
{
	//todo assert images not null

	auto image = images.Image;
	auto mask = images.Mask;
	auto backgroundMask = images.BackgroundMask;

	
	Mat maskDescriptors;
	ProcessImage(image, mask, maskDescriptors);
	Mat backgroundMaskDescriptors;
	ProcessImage(image, backgroundMask, backgroundMaskDescriptors);	

	Mat descriptors;
	descriptors.push_back(maskDescriptors);
	descriptors.push_back(backgroundMaskDescriptors);

	Mat labelsToReturn = CalculateLabels(maskDescriptors, backgroundMaskDescriptors);

	svmInput.Data = descriptors;
	svmInput.Labels = labelsToReturn;	
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
