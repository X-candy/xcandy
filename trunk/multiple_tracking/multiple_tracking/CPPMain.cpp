#include "common.h"
#include "function.h"



int main(int argc,char** argv)
{
	VideoCapture _capture;
	_capture.open("D:\\ABC.avi");
	Mat frame;
	if(!_capture.isOpened())
		return -1;

	/*char fData[2][3]={1,4,
	4,1,
	2,4};

	Mat abc = Mat(2,3,CV_8UC1,fData);
	cout<<"abc:"<<abc<<endl;
	Mat T = Mat(2,3,CV_8UC1);
	T.setTo(0);
	cout<<"T:"<<T<<endl;
	Mat A=abc.colRange(2,3);
	Mat Bc= T.colRange(2,3);

	A.copyTo(Bc);
	cout<<"T:"<<T<<endl;*/
	//for(int c=0;c<abc.channels();c++)
	//{
	//	for(int i=0;i<abc.rows;i++)
	//	{
	//		for(int j=0;j<abc.cols;j++)
	//		{
	//			cout<<abc.at<UCHAR>(i,j,c)<<endl;
	//		}
	//	}
	//	cout<<"***************"<<endl;
	//}
	////char fData[1]={0};
	////Mat abc = Mat(1,1,CV_8SC1,fData);
	////cout<<abc<<endl;
	//Mat b;

	////DiffMat(abc,b);
	////cout<<b<<endl;

	//nonunique(abc,b);

	//Mat ccv=abc.mul(abc);
	//cout<<ccv<<endl;
	//for(int i=0;i<2;i++)
	//{
	//	for(int k=0;k<4;k++)
	//	{
	//		printf("%d\t",abc.at<char>(i,k));
	//	}
	//	printf("\n");
	//}

	//printf("***************************************\n");
	//for(int i=0;i<2;i++)
	//{
	//	for(int k=0;k<4;k++)
	//	{
	//		printf("%d\t",abc.data[i*abc.step+k]);
	//	}
	//	printf("\n");
	//}

	//Scalar x= sum(abc);

	//Mat A = Mat::eye(3,3,CV_32F);
	//Mat B;
	////cv::sort(A, B, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
	//cv::sortIdx(A, B, CV_SORT_EVERY_ROW | CV_SORT_ASCENDING);
	//std::cout << A << "\n";
	//std::cout << B << "\n";
	//for(int i=0;i<3;i++)
	//{
	//	for(int k=0;k<3;k++)
	//	{
	//		printf("%f\t",A.at<float>(i,k));
	//	}
	//	printf("\n");
	//}
	//printf("*********************\n");
	//for(int i=0;i<3;i++)
	//{
	//	for(int k=0;k<3;k++)
	//	{
	//		printf("%d\t",B.at<UINT>(i,k));
	//	}
	//	printf("\n");
	//}
	//printf("*********************\n");


	vector<DATASET> dataset_info;
	char dataset_name[20]="slalom";
	
	char file_path[255];
	sprintf_s(file_path,"f:\\smot\\cdicle-smot-7ce1e201cf2d\\smot_data\\%s\\%s.itl",dataset_name,dataset_name);
	vector<DETECTRECT> detect_rect;
	read_dataset(file_path, dataset_info);
	ProcessDataSet(detect_rect,dataset_info);
	vector<Mat> B;
	vector<Mat> distanceSQ;
	findAssociations(detect_rect,3,B,distanceSQ);
	vector<I_TRACK_LINK> itl;
	linkDetectionTracklets(detect_rect,B,distanceSQ,itl);
	char file_name[255]={0};
	for(int i=0;i<detect_rect.size();i++)
	{		
		sprintf_s(file_name,"f:\\smot\\cdicle-smot-7ce1e201cf2d\\smot_data\\%s\\img\\img%05d.jpg",dataset_name,i+1);
		frame =imread(file_name);
		for(int k=0;k<detect_rect[i].detect_rect.size();k++)
		{
			Rect o_rect=Rect();
			o_rect.x = (int)detect_rect[i].detect_rect[k].x;
			o_rect.y = (int)detect_rect[i].detect_rect[k].y;
			o_rect.width = (int)detect_rect[i].detect_rect[k].width;
			o_rect.height = (int)detect_rect[i].detect_rect[k].height;
			rectangle(frame,o_rect,Scalar(0,0,255/detect_rect[i].detect_rect.size()*k),3);
		}
		imshow("frame",frame);
		waitKey(20);
	}
	while(1)
	{
		_capture>>frame;
		if(frame.empty())                                                                                             
			break;
		cv::imshow("frame",frame);
		cv::waitKey(20);
	}
}
