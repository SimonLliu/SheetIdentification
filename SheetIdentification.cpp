
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/opencv.hpp>
#include <imgproc/imgproc.hpp>
#include <stdlib.h>

using namespace cv;
Mat transformmat(Mat image, std::vector<cv::Point> docCnt);
Mat FourCorner(std::string a);
Mat sauvola(const cv::Mat &gray_scale, double k, int kernerl_width);
Mat Augment(Mat paper, Mat binary);
int main()
{
	for (int i = 1; i < 185; i++) {
		std::cout << i << std::endl;
		Mat paper, gray;
		std::string a = std::to_string(i);
		paper = FourCorner(a);
		std::string path0 = "E:\\com2_sau\\";
		path0 += a;
		path0 += "paper.jpg";
		imwrite(path0, paper);

		cvtColor(paper, gray, CV_RGB2GRAY);
		double k = 0.05;
		int kernerl_width = 500;
		Mat binary = sauvola(gray, k, kernerl_width);
		std::string path= "E:\\com2_sau\\";
		path += a;
		path += "binary.jpg";
		imwrite(path, binary);

		Mat comp = Augment(paper, binary);
		std::string path2 = "E:\\com2_sau\\";
		path2 += a;
		path2 += "comp.jpg";
		imwrite(path2, comp);

	}
	return 0;
}
Mat transformmat(Mat image, std::vector<cv::Point> docCnt)
{
	Mat result;
	Point2f srcTri[4], dstTri[4];
	Point2f docCnt_2[4];
	int x_half = image.cols / 2;
	int y_half = image.rows / 2;
	for (int i = 0; i < 4; i++){
		if (docCnt[i].x < x_half && docCnt[i].y < y_half) docCnt_2[0]=docCnt[i];
		if (docCnt[i].x > x_half && docCnt[i].y < y_half) docCnt_2[1]=docCnt[i];
		if (docCnt[i].x < x_half && docCnt[i].y > y_half) docCnt_2[2]=docCnt[i];
		if (docCnt[i].x > x_half && docCnt[i].y > y_half) docCnt_2[3]=docCnt[i];
	}
	int high_x = 0;
	int low_x = 0;
	int high_y = 0;
	int low_y = 0;
	int temp;
	for (int i = 0; i < 4;  i++) {
		if (docCnt[i].x > high_x) high_x = docCnt[i].x;
		if (docCnt[i].x < low_x) low_x = docCnt[i].x;
	}
	for (int i = 0; i < 4; i++) {
		if (docCnt[i].y > high_y) high_y = docCnt[i].y;
		if (docCnt[i].y < low_y) low_y = docCnt[i].y;
	}
	int height = high_y - low_y;
	int width = high_x - low_x;
	dstTri[0].x = 0;
	dstTri[0].y = 0;
	dstTri[1].x = image.cols;
	dstTri[1].y = 0;
	dstTri[2].x = 0;
	dstTri[2].y = image.rows;
	dstTri[3].x = image.cols;
	dstTri[3].y = image.rows;
	srcTri[0].x = docCnt_2[0].x;
	srcTri[0].y = docCnt_2[0].y;
	srcTri[1].x = docCnt_2[1].x;
	srcTri[1].y = docCnt_2[1].y;
	srcTri[2].x = docCnt_2[2].x;
	srcTri[2].y = docCnt_2[2].y;
	srcTri[3].x = docCnt_2[3].x;
	srcTri[3].y = docCnt_2[3].y;
	Mat transform = Mat::zeros(3, 3, CV_32FC1); //透视变换矩阵
	transform = getPerspectiveTransform(srcTri, dstTri);  //获取透视变换矩阵	
	//warpPerspective(image, result, transform, Size(image.cols + 1, image.rows + 1));
	warpPerspective(image, result, transform, Size(image.cols, image.rows));
	return result;
}
Mat FourCorner(std::string a)
{
	std::string str = "E:\\jpg\\";
	str += a;
	str += ".jpg";
	Mat image = imread(str);
	//namedWindow("ͼƬ");
	//imshow("ͼƬ", imag);   
	//waitKey(12000);
	Mat image2;
	image.copyTo(image2);
	double j = 0, k;
	Mat gray, blurred, edged;
	int z;
	int count = 0;
	std::vector<std::vector<cv::Point>>cnts;
	std::vector<cv::Point>c;
	std::vector<cv::Point> docCnt;
	//vector<Vec4i>hierarchy;
	for (j; j < 8.0; j++) {
		if (docCnt.empty()) {
			cvtColor(image2, gray, CV_RGB2GRAY);
			for (z = 43,count=0; count < 2; z = 33) {
				count++;
				medianBlur(gray, blurred, z);
				adaptiveThreshold(blurred, blurred, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 51, 2);
				Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
				dilate(blurred, blurred, kernel);
				Canny(blurred, edged, 10, 100);
				findContours(edged, cnts, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
				//cnts = cnts[0];
				//Mat result(imag.size(), CV_8U, cv::Scalar(255));
				//drawContours(result, cnts, -1, cv::Scalar(0), 2);
				//namedWindow("resultImage", 1);
				//imshow("resultImage", result);
				//if (cnts.size() > 0) {
				//	printf("%d", cnts.size());
				//}
				if (cnts.size() > 0) {

					std::vector<double>temp_area;
					double temp = 0;
					double area = 0;
					for (int i = 0; i < cnts.size(); i++) {
						temp_area.push_back(contourArea(cnts[i]));
						if (temp_area[i] > temp) {
							temp = temp_area[i];
							c = cnts[i];
							area = temp_area[i];
						}
					}
					double peri = arcLength(c, true);
					std::cout << area << std::endl;
					if (area >= 2000000) {
						std::vector<cv::Point> approx;
						for (double a = 1; a < 10; a++) {
							approxPolyDP(c, approx, 0.1*a*peri, true);
							if (approx.size() == 4) {
								docCnt = approx;
								break;
							}
						}
					}
				}
				if (!docCnt.empty()) break;
			}
		}
		if (docCnt.empty() && j == 7) {
			k = 10;
			break;
		}
		if (!docCnt.empty()) {
			k = j;
			break;
		}
		pyrMeanShiftFiltering(image, image2, 5, ((j + 1) * 10));
	}
	Mat paper;
	if (k != 10) {
		paper = transformmat(image, docCnt);
	}
	else paper = image;
	/*for (int u = 0; u < 4; u++) {
		printf("%d %d\n", docCnt[u].x, docCnt[u].y);
	}*/
	//char l;
	//std::cin >> l;
	/*for (int i = 0; i < 4; i++) {
		circle(image, docCnt[i], 50, (255, 0, 0), -1);
	}*/
	
	//imwrite("E:\\jpg\\210_0_3.jpg", image);
	return paper;

}
Mat sauvola(const cv::Mat &gray_scale, double k, int kernerl_width)      //k通常为0.几，kernerl_width通常为几十
{
	cv::Mat integral_sum, integral_sqrSum;
	cv::Mat mean = cv::Mat::zeros(gray_scale.size(), CV_32F);
	cv::Mat threshold = cv::Mat::zeros(gray_scale.size(), CV_8UC1);
	cv::Mat diff = cv::Mat::zeros(gray_scale.size(), CV_32F);
	cv::Mat sqrDiff = cv::Mat::zeros(gray_scale.size(), CV_32F);
	cv::Mat Std = cv::Mat::zeros(gray_scale.size(), CV_32F);
	cv::Mat binary_image = cv::Mat::zeros(gray_scale.size(), CV_8UC1);
	integral(gray_scale, integral_sum, integral_sqrSum, CV_32F, CV_32F);
	cv::Mat integral_sum_tailored(integral_sum, cv::Rect(1, 1, gray_scale.cols, gray_scale.rows));
	cv::Mat integral_sqrSum_tailored(integral_sqrSum, cv::Rect(1, 1, gray_scale.cols, gray_scale.rows));
	int xmin, ymin, xmax, ymax, area, whalf = kernerl_width >> 1, height = gray_scale.rows, width = gray_scale.cols;


	for (int row = 0; row < integral_sum_tailored.rows; row++)
		for (int col = 0; col < integral_sum_tailored.cols; col++)
		{
			xmin = std::max(0, col - whalf);
			ymin = std::max(0, row - whalf);
			xmax = std::min(width - 1, col + whalf);
			ymax = std::min(height - 1, row + whalf);

			area = (xmax - xmin + 1) * (ymax - ymin + 1);
			if (area <= 0)
			{
				exit(1);
			}



			if (xmin == 0 && ymin == 0) {                            //假如Kernerl的左上角未离开图片左上角，则中点的diff和sqrDiff.at<float>等于
				diff.at<float>(row, col) = integral_sum_tailored.at<float>(ymax, xmax);       //Kernerl右下角的integralImag和integral_sum_tailored.at<float>Sqrt
				sqrDiff.at<float>(row, col) = integral_sqrSum_tailored.at<float>(ymax, xmax);
			}
			else if (xmin > 0 && ymin == 0) {
				diff.at<float>(row, col) = integral_sum_tailored.at<float>(ymax, xmax) - integral_sum_tailored.at<float>(ymax, xmin - 1);             //假如Kernerl上端未离开，则“中点”的diff、sqrDiff.at<float>等于
				sqrDiff.at<float>(row, col) = integral_sqrSum_tailored.at<float>(ymax, xmax) - integral_sqrSum_tailored.at<float>(ymax, xmin - 1);   //                                  Kernerl右下角-左下角
			}
			else if (xmin == 0 && ymin > 0) {
				diff.at<float>(row, col) = integral_sum_tailored.at<float>(ymax, xmax) - integral_sum_tailored.at<float>(ymin - 1, xmax);            //假如Kernerl左端未离开，则“中点”的diff、sqrDiff.at<float>等于
				sqrDiff.at<float>(row, col) = integral_sqrSum_tailored.at<float>(ymax, xmax) - integral_sqrSum_tailored.at<float>(ymin - 1, xmax);; //                        Kernerl右下角-右上角
			}
			else {
				float diagsum = integral_sum_tailored.at<float>(ymax, xmax) + integral_sum_tailored.at<float>(ymin - 1, xmin - 1);       //假如Kernerl左和上完全离开，则中点的diff、sqrDiff.at<float>等于
				float idiagsum = integral_sum_tailored.at<float>(ymin - 1, xmax) + integral_sum_tailored.at<float>(ymax, xmin - 1);      //主对角线两元素和-副对角线两元素和
				diff.at<float>(row, col) = diagsum - idiagsum;

				float sqdiagsum = integral_sqrSum_tailored.at<float>(ymax, xmax) + integral_sqrSum_tailored.at<float>(ymin - 1, xmin - 1);
				float sqidiagsum = integral_sqrSum_tailored.at<float>(ymin - 1, xmax) + integral_sqrSum_tailored.at<float>(ymax, xmin - 1);
				sqrDiff.at<float>(row, col) = sqdiagsum - sqidiagsum;
			}

			mean.at<float>(row, col) = diff.at<float>(row, col) / area;
			Std.at<float>(row, col) = sqrt((sqrDiff.at<float>(row, col) - diff.at<float>(row, col) * diff.at<float>(row, col) / area) / (area - 1));
			threshold.at<uchar>(row, col) = mean.at<float>(row, col) * (1 + k * ((Std.at<float>(row, col) / 128) - 1));
			if (gray_scale.at<uchar>(row, col) < threshold.at<uchar>(row, col))

				binary_image.at<uchar>(row, col) = 255;
			else
				binary_image.at<uchar>(row, col) = 0;
		}
	return binary_image;
}
Mat Augment(Mat paper, Mat binary)
{
	Mat image_yuv,output,paper2;
	cvtColor(paper, image_yuv, COLOR_BGR2YUV);
	std::vector<Mat> channels;
	split(image_yuv, channels);
	equalizeHist(channels.at(0), channels.at(0));
	merge(channels, paper2);
	cvtColor(paper2, output, COLOR_YUV2BGR);
	Mat eh;
	//output.copyTo(eh);
	eh = output.clone();
	int rows = eh.rows;
	int cols = eh.cols;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (binary.at<uchar>(i, j) == 0) {
				eh.at<Vec3b>(i, j)[0] = 255;//blue 通道
				eh.at<Vec3b>(i, j)[1] = 255;//green 通道
				eh.at<Vec3b>(i, j)[2] = 255;//red 通道
			}
		}
	}
	return eh;

}