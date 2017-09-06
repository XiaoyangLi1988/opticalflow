// opticalflow.cpp : enterance��
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cstdio>
using namespace std;
using namespace cv;

bool verifySizes(Rect R){

    float aspect=1.5;

    int min= 120*aspect*120; // minimum area

    int max= 200*aspect*200; // maximum area
    //Get only patchs that match to a respect ratio.

    int area= R.height * R.width;

    if(( area < min || area > max )){
        return false;
    }else{
        return true;
    }

}
 
void tracking(Mat &frame, Mat &output);
 
bool addNewPoints0( );

bool addNewPoints1( );
 
bool acceptTrackedPoint(int i);

bool acceptTrackedPointa(int i);
 

string window_name = "motion detection";
 
Mat gray;        // current frame
 
Mat gray_prev;        // predict frame
 
vector<Point2f> points[4];        // point0Ϊ�������ԭ��λ�ã�point1Ϊ���������λ��
 
vector<Point2f> initial,initial1;        // ��ʼ�����ٵ��λ��
 
vector<Point2f> features;        // ��������
 
int maxCount = 500;        // �������������
 
double qLevel = 0.01;        // �������ĵȼ�
 
double minDist = 10.0;        // ��������֮�����С����
 
vector<uchar> status,status1;        // ����������״̬��������������Ϊ1������Ϊ0
 
vector<float> err,err1;
 

int main()
 
{
 
        Mat frame,grayframe,grayframe1,binaryframe,frame1;
 
        Mat result;

		 Mat dilatelement = getStructuringElement(MORPH_RECT, Size(13,13));//�����ں�
 

//         CvCapture* capture = cvCaptureFromCAM( -1 );        // ����ͷ��ȡ�ļ�����
 
        VideoCapture capture("test.mp4");
 

        if(capture.isOpened()/*capture*/)        // ����ͷ��ȡ�ļ�����
 
        {
 
                while(true)
 
                {
 
                           capture >> frame;

						   capture>>frame1;

                        if(!frame.empty()&&!frame1.empty())
 
                        { 
                                tracking(frame, result);

								cvtColor(frame,grayframe,CV_BGR2GRAY);

								cvtColor(frame1,grayframe1,CV_BGR2GRAY);

							   absdiff(grayframe,grayframe1,binaryframe);

							   threshold(binaryframe,binaryframe,0,255,CV_THRESH_OTSU);

							   dilate(binaryframe,binaryframe, dilatelement);

							   vector<vector<Point>>  contours;

							   findContours(binaryframe,contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); 

	                           vector<vector<Point> >::iterator itc= contours.begin();

							   vector<Rect> rects;

								
								while (itc!=contours.end()) {

									Rect r = boundingRect(Mat(*itc));

									if( !verifySizes(r)){

										itc= contours.erase(itc);
									}
									else{
										   ++itc;

										   rects.push_back(r);
									}
								}

								for(int j = 0;j<rects.size();j ++)
									rectangle(result,rects[j],Scalar(0,0,255),2);

								imshow(window_name, result);//display

                        }
 
                        else
 
                        { 
                                printf("can not open��break");
 
                                break;
 
                        }
 

                        int c = waitKey(20);
 
                        if( (char)c == 27 )
 
                        {
 
                                break; 
                        } 
                }
 
        }
 
        return 0;
 
}
 

//////////////////////////////////////////////////////////////////////////
 
// function: tracking
 
// brief: ����
 
// parameter: frame        �������Ƶ֡
 
//                          output �и��ٽ������Ƶ֡
 
// return: void
 
//////////////////////////////////////////////////////////////////////////
 
void tracking(Mat &frame, Mat &output)
 
{
 
        cvtColor(frame, gray, CV_BGR2GRAY);
 
        frame.copyTo(output);
 
        // ���������
 
        if (addNewPoints0())
 
        {
 
                goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist);   
 
                points[0].insert(points[0].end(), features.begin(), features.end());
 
                initial.insert(initial.end(), features.begin(), features.end());
 
        }

		if (addNewPoints1())
 
        {
 
                goodFeaturesToTrack(gray, features, maxCount, qLevel, minDist);   
 
                points[2].insert(points[2].end(), features.begin(), features.end());
 
                initial1.insert(initial1.end(), features.begin(), features.end());
 
        }
 

        if (gray_prev.empty())
 
        {
 
                gray.copyTo(gray_prev);
 
        }
 
        // l-k�������˶�����
 
        calcOpticalFlowPyrLK(gray_prev, gray, points[0], points[1], status, err);

		calcOpticalFlowPyrLK(gray, gray_prev, points[2], points[3], status1, err1);


 
        // ȥ��һЩ���õ�������
 
        int k = 0;
 
        for (size_t i=0; i<points[1].size(); i++)
 
        {
 
                if (acceptTrackedPoint(i))
 
                {
 
                        initial[k] = initial[i];
 
                        points[1][k++] = points[1][i];
 
                }
 
        }
 
        points[1].resize(k);
 
        initial.resize(k);

		

        int k1 = 0;
 
        for (size_t i=0; i<points[3].size(); i++)
 
        {
 
                if (acceptTrackedPointa(i))
 
                {
 
                        initial1[k1] = initial1[i];
 
                        points[3][k1++] = points[3][i];
 
                }
 
        }
 
        points[3].resize(k1);
 
        initial1.resize(k1);
 
        // ��ʾ��������˶��켣
 
        for (size_t i=0; i<points[1].size(); i++)
 
        {
 
                line(output, initial[i], points[1][i], Scalar(0, 0, 255));
 
                circle(output, points[1][i], 2, Scalar(255, 0, 0), -1);
 
        }

		 // ��ʾ��������˶��켣
 
        for (size_t i=0; i<points[3].size(); i++)
 
        {
 
                line(output, initial1[i], points[3][i], Scalar(0, 255, 0));
 
                circle(output, points[3][i], 2, Scalar(0, 255, 255), -1);
 
        }
 

        // �ѵ�ǰ���ٽ����Ϊ��һ�˲ο�
 
        swap(points[1], points[0]);

		swap(points[3], points[2]);
 
        swap(gray_prev, gray);
 
 
}
 

//////////////////////////////////////////////////////////////////////////
 
// function: addNewPoints
 
// brief: ����µ��Ƿ�Ӧ�ñ����
 
// parameter:
 
// return: �Ƿ���ӱ�־
 
//////////////////////////////////////////////////////////////////////////
 
bool addNewPoints0()
 
{
 
        return points[0].size() <= 20;
 
}

bool addNewPoints1()
 
{
 
        return points[2].size() <= 20;
 
}
 

//////////////////////////////////////////////////////////////////////////
 
// function: acceptTrackedPoint
 
// brief: ������Щ���ٵ㱻����
 
// parameter:
 
// return:
 
//////////////////////////////////////////////////////////////////////////
 
bool acceptTrackedPoint(int i)
 
{
 
        return  status[i] && (( norm(points[1][i] - points[0][i])) >=2);
 
}

bool acceptTrackedPointa(int i)
 
{
 
        return  status1[i] && (( norm(points[2][i] - points[3][i])) >=2);
 
}
