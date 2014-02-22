#include "Tracker.h"


CTracker::CTracker(void)
{
	m_dSimilarity_method=1;
	m_nHor = 80;
	m_ratio_threhold =3;
	m_dMin_similarity=0.01;
	m_dMota_threhold=0.5;
	m_dEta_max = 3;
	m_bDebug=1;
	m_bQcheck=false;
	m_dGap=0;
	m_dHor_max = INF;
	m_dGap_max = INF;
	m_dSlope_max= INF;
	m_lastminD2=0;

	m_nHistoryLen= 15;
	m_detect_rect_squence.resize(m_nHistoryLen);
	//m_B.resize(m_nHistoryLen);
	//m_distanceSQ.resize(m_nHistoryLen);
}


CTracker::~CTracker(void)
{
}

int CTracker::tracker(int _frame_num,vector<Rect> _detect_rect,Mat _frame)
{
	m_frame_num = _frame_num;
	_frame.copyTo(m_frame);
	for(int i=0;i<_detect_rect.size();i++)
	{
		rectangle(m_frame,_detect_rect[i],Scalar(255,0,0));
	}

	InputDetectRect(m_detect_rect_squence,_frame_num,_detect_rect);
	FindAssociations(m_detect_rect_squence,m_ratio_threhold,m_B,m_distanceSQ);
	cout<<"********"<<_frame_num<<"*********"<<endl;
	if(_frame_num >70)
	{
		for(int i=0;i<m_B.size();i++)
		{
			cout<<m_B[i];
		}
		cout<<endl<<"*****************"<<endl;
	}
	
	///*if(_frame_num>30)*/


	//11:22 -2014/2/21临时注释

//	LinkDetectionTracklets(m_detect_rect_squence,m_B,m_distanceSQ,m_itl);
	//LinkDetectionTracklets1(m_detect_rect_squence,m_B,m_distanceSQ,m_itl1);


	//if(m_frame_num<100 && m_frame_num>71)
	//{
	//	cout<<"**********************m_itl*************************"<<endl;
	//	for(int i=0;i<m_itl.size();i++)
	//	{
	//		cout<<m_itl[i].xy_data<<endl;
	//	}
	//	cout<<"**********************m_itl*************************"<<endl;
	//	for(int i=0;i<m_itl1.size();i++)
	//	{
	//		cout<<m_itl1[i].xy_data<<endl;
	//	}
	//}

	//if(m_itl.size()>=1)
	//{
	//	for(int i=0;i< m_itl.size();i++)
	//	{
	//		if(m_itl[i].length>=3)
	//		{
	//			for(int j=0;j<m_itl[i].length-1;j++)
	//			{
	//				Point x1;
	//				x1.x = (int)m_itl[i].xy_data.at<float>(0,j);
	//				x1.y = (int)m_itl[i].xy_data.at<float>(1,j);
	//				cv::circle( m_frame, x1, 1, cv::Scalar( i*20&255 , i*20&255 , 0 ), 3 );
	//			}
	//			imshow("m_frame",m_frame);
	//		}
	//	}

	//}
	//waitKey(1);
	//int nStartFrame=0;
	//int nEndFrame=m_nHistoryLen-1;
	//if(m_itl.size()>0)
	//{
	//	nStartFrame = m_itl[0].t_start;
	//	nEndFrame = m_itl[0].t_end;
	//}


	//for(int i=0;i<m_itl.size();i++)
	//{
	//	if(nStartFrame > m_itl[i].t_start)
	//		nStartFrame = m_itl[i].t_start;
	//	if(nEndFrame < m_itl[i].t_end)
	//		nEndFrame = m_itl[i].t_end;
	//}

	//Associate_ITL(m_itl,nStartFrame,nEndFrame);
	return 1;
}

int CTracker::InputDetectRect(vector<DETECTRECT> &_detect_rect_squence,int _frame_num,vector<Rect> _detect_rect)
{
	//队列不满
	if(_frame_num < m_nHistoryLen)
	{
		_detect_rect_squence[_frame_num].detect_rect=_detect_rect;
		for (int i=0;i<_detect_rect.size();i++)
		{
			//生成rect的IDX
			_detect_rect_squence[_frame_num].idx.push_back(i);
			Point center;
			center.x = _detect_rect_squence[_frame_num].detect_rect[i].x + _detect_rect_squence[_frame_num].detect_rect[i].width/2;
			center.y = _detect_rect_squence[_frame_num].detect_rect[i].y + _detect_rect_squence[_frame_num].detect_rect[i].height/2;
			_detect_rect_squence[_frame_num].detect_rect_center.push_back(center);
			_detect_rect_squence[_frame_num].object_id.push_back(0);
		}
		_detect_rect_squence[_frame_num].frame_num= _frame_num;
	}
	//队列满，擦处队列第一个
	else
	{
		_detect_rect_squence.erase(_detect_rect_squence.begin());
		DETECTRECT tempDetectRect;
		tempDetectRect.detect_rect = _detect_rect;
		for (int i=0;i<_detect_rect.size();i++)
		{
			//生成rect的IDX
			tempDetectRect.idx.push_back(i);
			Point center;
			center.x = tempDetectRect.detect_rect[i].x + tempDetectRect.detect_rect[i].width/2;
			center.y = tempDetectRect.detect_rect[i].y + tempDetectRect.detect_rect[i].height/2;
			tempDetectRect.detect_rect_center.push_back(center);
			tempDetectRect.object_id.push_back(0);
		}
		tempDetectRect.frame_num = _frame_num;

		_detect_rect_squence.push_back(tempDetectRect);
	}
	return 1;
}

void  CTracker::DiffMat(Mat _a,Mat &_b)  //求向量B的一阶差分 功能等价matlab里的diff
{
	int cols = _a.cols;
	int rows = _a.rows;
	int nChannels = _a.channels();
	if(rows!=1)
		_a=_a.reshape(_a.channels(),1);
	_a.convertTo(_a,CV_32F);
	if(_b.empty())
		_b=Mat(1,cols-1,CV_32F);
	float *pB = (float*)(_a.data);
	float *pOut =(float*)(_b.data);
	for(int i=0; i<cols-1; i++)
	{
		*pOut = *(pB+1)-*pB;
		pB++;
		pOut++;
	}
}

void CTracker::NONUnique(Mat _a,Mat _distance,Mat &_b)
{
	if(m_frame_num >=91)
		int klkl=0;
	if(_a.empty())
		return;
	int rows = _a.rows;
	int cols = _a.cols;

	int rowvec = (rows == 1) &&(cols>1);
	//矩阵元素总数
	int numelA = cols * rows;
	if(numelA == 1)
	{
		_a.copyTo(_b);
		_b.convertTo(_b,CV_32F);
		return;
	}
	//cout<<_b<<endl;
	Mat tempA;
	tempA=_a.reshape(1,1);
	Mat sorted;
	Mat sortedIdx;
	tempA.convertTo(tempA,CV_32F);
	cv::sort(tempA,sorted,CV_SORT_ASCENDING);
	cv::sortIdx(tempA,sortedIdx,CV_SORT_ASCENDING);

	Mat db;
	DiffMat(sorted,db);
	cout<<"db="<<db<<endl;
	Mat d = Mat(1,cols*rows,CV_8U);
	Mat tempd= (db!=0);
	cout<<"tempd="<<tempd<<endl;
	int tempd_size=tempd.channels()*tempd.cols*tempd.step*tempd.elemSize();
	memcpy(d.data,tempd.data,tempd_size);
	d.at<uchar>(0,cols*rows-1)=255;

	int nNonZero=countNonZero(d);
	int nZero = d.cols * d.rows - nNonZero;
	//Mat
	//for(int i=0;i<d.cols * d.rows;i++)
	//{
	//	if()
	//}

	Mat tempB=Mat(1,nNonZero,CV_32F);

	int k=0;
	for(int i=0;i<rows*cols;i++)
	{
		if(d.at<uchar>(0,i) == 255)
		{
			if(k<nNonZero)
			{
				tempB.at<float>(0,k)=sorted.at<float>(0,i);
				k++;
			}
		}
	}

	if(nNonZero ==1 && numelA >1)
	{
		tempB.copyTo(_b);
	}
	else
	{
		//计算矩阵中各值的直方图
		vector<int> nhistCount(nNonZero);

		for(int k=0;k<numelA;k++)
		{
			for(int i=0;i<nNonZero;i++)
			{
				if(tempB.at<float>(0,i) == tempA.at<float>(0,k))
					nhistCount[i]++;
			}
		}

		//重复元素进行压栈
		vector<float> tempVectorB;
		for(int k=0;k<nNonZero;k++)
		{
			if(nhistCount[k]>1)
			{
				tempVectorB.push_back(tempB.at<float>(0,k));
			}
		}

		if(tempVectorB.size() != 0)
		{
			for(int k=0;k<tempVectorB.size();k++)
			{
				if(tempVectorB[k] > 0)
				{
					double temp_distance=-1;
					for(int j=0;j<numelA;j++)
					{			
						//重复元素的处理
						if(tempA.at<float>(0,j) == tempVectorB[k] )
						{
							temp_distance = _distance.at<float>(j,0);
							//tempA.at<float>(0,j) = tempVectorB[k];
							//距离远的置为-1，距离近的置为下一序
							
							if(temp_distance<)
							{

							}
							else
							{
								tempA.at<float>(0,j) = tempVectorB[k];
							}
						}
					}
				}
			}
		}
		tempA.copyTo(_b);
		cout<<"tempA="<<tempA<<endl;
	}
}

int CTracker::CalRectDistance(DETECTRECT _detect_rect_t,DETECTRECT _detect_rect_tp1,int _ratio_threhold,Mat &_mat_distance,Mat &_rlt)
{
	int N_t=_detect_rect_t.detect_rect.size();
	//t+1时刻的检测框数量
	int N_tp1=_detect_rect_tp1.detect_rect.size();
	if( N_t > 0 && N_tp1> 0)
	{
		Mat mat_distance = Mat(N_t,N_tp1,CV_32FC1);
		//计算检测框间的距离
		for(int NT_i=0 ; NT_i< N_t ;NT_i++)
		{
			for(int NT_ip1=0;NT_ip1<N_tp1;NT_ip1++)
			{
				mat_distance.at<float>(NT_i,NT_ip1)=\
					pow((double)(_detect_rect_t.detect_rect_center[NT_i].x - _detect_rect_tp1.detect_rect_center[NT_ip1].x),2)\
					+pow((double)(_detect_rect_t.detect_rect_center[NT_i].y- _detect_rect_tp1.detect_rect_center[NT_ip1].y),2);
				mat_distance.at<float>(NT_i,NT_ip1) = sqrt(mat_distance.at<float>(NT_i,NT_ip1));
			}
		}
		mat_distance.copyTo(_mat_distance);

		//cout<<"_mat_distance="<<_mat_distance<<endl;
		//检测前两位距离比值
		Mat Dsorted;
		cv::sort(mat_distance,Dsorted,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING );
		//cout<<"Dsorted="<<Dsorted<<endl;
		Mat DIdx;
		cv::sortIdx(mat_distance,DIdx,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING );
		//cout<<"DIdx="<<DIdx<<endl;
		if(N_tp1>1)
		{
			Mat A=Dsorted.colRange(0,1);
			Mat B=Dsorted.colRange(1,2);
			Mat C;
			divide(B,A,m_Ratio);
			m_f=m_Ratio>_ratio_threhold;
			for(int i=0;i<A.rows;i++)
			{
				if(A.at<float>(i,0)==0)
					m_f.at<uchar>(i,0) = 255;
			}
			minMaxLoc(Dsorted,NULL,&m_lastminD2,NULL,NULL);    // 不需要的置为0 
		}
		else if(m_lastminD2>0)
		{
			Mat A=Dsorted.colRange(0,1);
			divide(m_lastminD2,A,m_Ratio);
			m_f=m_Ratio>_ratio_threhold;
			for(int i=0;i<A.rows;i++)
			{
				if(A.at<float>(i,0)==0)
					m_f.at<uchar>(i,0) = 255;
			}
		}
		else
		{
			Mat A=Dsorted.colRange(0,1);
			m_Ratio = A*0;
			m_f= m_Ratio>_ratio_threhold;
		}

		Mat A=DIdx.colRange(0,1);
		A.convertTo(A,CV_8UC1);

		Mat tempmat=A.mul(m_f/255);
		Mat rlt;
		cout<<"DIdx="<<tempmat<<endl;
		//检测重复的idx
		NONUnique(tempmat,rlt,mat_distance);
		
		cout<<"rlt="<<rlt<<endl;
		rlt.copyTo(_rlt);
	}
	else
	{
		return -1;
	}
	return 1;
}

int CTracker::FindAssociations(vector<DETECTRECT> &_detect_rect_squence,int _ratio_threhold,vector<Mat> &_b,vector<Mat> &_distance)
{
	//序列不满足m_nHistoryLen长度的
	if(m_frame_num >= m_nHistoryLen )
	{
		_b.erase(_b.begin());
		_distance.erase(_distance.begin());
	}


	if(m_frame_num > 0)
	{
		int T=_b.size();
		//上一时刻的检测框数量
		int N_t=_detect_rect_squence[T].detect_rect.size();
		//当前时刻的检测框数量
		int N_tp1=_detect_rect_squence[T-1].detect_rect.size();
		if( N_t > 0 && N_tp1> 0)
		{
			Mat mat_distance = Mat(N_t,N_tp1,CV_32FC1);
			Mat rlt;
			CalRectDistance(_detect_rect_squence[T],_detect_rect_squence[T-1],_ratio_threhold,mat_distance,rlt);
			//CalRectDistance(_detect_rect_squence[T],_detect_rect_squence[T+1],_ratio_threhold,mat_distance,rlt);
			_distance.push_back(mat_distance);
			_b.push_back(rlt);	
			//cout<<"_distance["<<T<<"]="<<_distance[T]<<endl;
			cout<<"_b["<<T<<"]="<<_b[T]<<endl;
			/*
			for(int i=0;i<_detect_rect_squence[T].detect_rect.size();i++)
			{
			rectangle(m_frame,_detect_rect_squence[T].detect_rect[i],Scalar(255,0,0));
			}

			for(int i=0;i<rlt.cols;i++)
			{
			int indx= rlt.at<float>(i);
			rectangle(m_frame,_detect_rect_squence[T+1].detect_rect[indx],Scalar(0,255,0));
			}
			imshow("m_frame",m_frame);
			waitKey(1);*/
		}
		else
		{
			N_tp1 = _detect_rect_squence[T].detect_rect.size();
			Mat tempmat=Mat(1,N_tp1,CV_32FC1);
			tempmat.setTo(-1);
			_b.push_back(tempmat);

			Mat temp_distance=Mat(1,0,CV_32F);
			_distance.push_back(temp_distance);
		}
		if(N_t > N_tp1)
		{
			Mat tempmat = Mat(1,N_t,CV_32FC1);

		}
	}
	else if(m_frame_num ==0)
	{
		Mat tempmat=Mat(1,0,CV_32F);
		_b.push_back(tempmat);

		Mat temp_distance=Mat(1,0,CV_32F);
		_distance.push_back(temp_distance);
	}
	else
	{
		int T=_b.size()-1;
		Mat tempmat=Mat(1,0,CV_32F);
		tempmat.copyTo(_b[T]);

		Mat temp_distance=Mat(1,0,CV_32F);
		temp_distance.copyTo(_distance[T]);
	}
	//else if(m_frame_num >= 2)
	//{
	//	//求取倒数第二帧的距离矩阵
	//	int T=m_nHistoryLen-2;
	//	int N_t=_detect_rect_squence[T].detect_rect.size();
	//	//t+1时刻的检测框数量
	//	int N_tp1=_detect_rect_squence[T+1].detect_rect.size();
	//	if( N_t > 0 && N_tp1> 0)
	//	{
	//		Mat mat_distance = Mat(N_t,N_tp1,CV_32FC1);
	//		Mat rlt;
	//		CalRectDistance(_detect_rect_squence[T],_detect_rect_squence[T+1],_ratio_threhold,mat_distance,rlt);

	//		//copy到对应的序列中
	//		mat_distance.copyTo(_distance[T-1]);
	//		rlt.copyTo(_b[T-1]);

	//		cout<<"_distance["<<T<<"]="<<_distance[T]<<endl;
	//		cout<<"_b["<<T<<"]="<<_b[T]<<endl;
	//		for(int i=0;i<_detect_rect_squence[T].detect_rect.size();i++)
	//		{
	//			rectangle(m_frame,_detect_rect_squence[T].detect_rect[i],Scalar(255,0,0));
	//		}

	//		for(int i=0;i<rlt.cols;i++)
	//		{
	//			int indx= rlt.at<float>(i);
	//			rectangle(m_frame,_detect_rect_squence[T+1].detect_rect[indx],Scalar(0,255,0));
	//		}
	//		imshow("m_frame",m_frame);

	//	}
	//	//序列进行前移
	//	_distance.erase(_distance.begin());
	//	Mat temp_distance=Mat();
	//	_distance.push_back(temp_distance);
	//	_b.erase(_b.begin());
	//	N_tp1 = _detect_rect_squence[m_nHistoryLen-1].detect_rect.size();
	//	Mat tempmat=Mat(1,N_tp1,CV_32FC1);
	//	tempmat.setTo(-1);
	//	_b.push_back(tempmat);
	//}
	
	return 1;
}

//分段程序
//int CTracker::FindAssociations(vector<DETECTRECT> &_detect_rect_squence,int _ratio_threhold,vector<Mat> &_b,vector<Mat> &_distance)
//{
//	//序列不满足m_nHistoryLen长度的
//	if(m_frame_num >= m_nHistoryLen )
//	{
//		_b.erase(_b.begin());
//		_distance.erase(_distance.begin());
//	}
//
//
//	if(m_frame_num >= 1)
//	{
//		int T=_b.size()-1;
//		//上一时刻的检测框数量
//		int N_t=_detect_rect_squence[T].detect_rect.size();
//		//当前时刻的检测框数量
//		int N_tp1=_detect_rect_squence[T+1].detect_rect.size();
//		if( N_t > 0 && N_tp1> 0)
//		{
//			Mat mat_distance = Mat(N_t,N_tp1,CV_32FC1);
//			Mat rlt;
//			CalRectDistance(_detect_rect_squence[T+1],_detect_rect_squence[T],_ratio_threhold,mat_distance,rlt);
//			//CalRectDistance(_detect_rect_squence[T],_detect_rect_squence[T+1],_ratio_threhold,mat_distance,rlt);
//			mat_distance.copyTo(_distance[T]);
//			rlt.copyTo(_b[T]);	
//	/*		cout<<"_distance["<<T<<"]="<<_distance[T]<<endl;
//			cout<<"_b["<<T<<"]="<<_b[T]<<endl;*/
//			/*
//			for(int i=0;i<_detect_rect_squence[T].detect_rect.size();i++)
//			{
//			rectangle(m_frame,_detect_rect_squence[T].detect_rect[i],Scalar(255,0,0));
//			}
//
//			for(int i=0;i<rlt.cols;i++)
//			{
//			int indx= rlt.at<float>(i);
//			rectangle(m_frame,_detect_rect_squence[T+1].detect_rect[indx],Scalar(0,255,0));
//			}
//			imshow("m_frame",m_frame);
//			waitKey(1);*/
//		}
//
//		N_tp1 = _detect_rect_squence[T+1].detect_rect.size();
//		Mat tempmat=Mat(1,N_tp1,CV_32FC1);
//		tempmat.setTo(-1);
//		_b.push_back(tempmat);
//
//		Mat temp_distance=Mat(1,0,CV_32F);
//		_distance.push_back(temp_distance);
//	}
//	else if(m_frame_num == 0)
//	{
//		Mat tempmat=Mat(1,0,CV_32F);
//		_b.push_back(tempmat);
//
//		Mat temp_distance=Mat(1,0,CV_32F);
//		_distance.push_back(temp_distance);
//	}
//	else
//	{
//		int T=_b.size()-1;
//		Mat tempmat=Mat(1,0,CV_32F);
//		tempmat.copyTo(_b[T]);
//
//		Mat temp_distance=Mat(1,0,CV_32F);
//		temp_distance.copyTo(_distance[T]);
//	}
//	//else if(m_frame_num >= 2)
//	//{
//	//	//求取倒数第二帧的距离矩阵
//	//	int T=m_nHistoryLen-2;
//	//	int N_t=_detect_rect_squence[T].detect_rect.size();
//	//	//t+1时刻的检测框数量
//	//	int N_tp1=_detect_rect_squence[T+1].detect_rect.size();
//	//	if( N_t > 0 && N_tp1> 0)
//	//	{
//	//		Mat mat_distance = Mat(N_t,N_tp1,CV_32FC1);
//	//		Mat rlt;
//	//		CalRectDistance(_detect_rect_squence[T],_detect_rect_squence[T+1],_ratio_threhold,mat_distance,rlt);
//
//	//		//copy到对应的序列中
//	//		mat_distance.copyTo(_distance[T-1]);
//	//		rlt.copyTo(_b[T-1]);
//
//	//		cout<<"_distance["<<T<<"]="<<_distance[T]<<endl;
//	//		cout<<"_b["<<T<<"]="<<_b[T]<<endl;
//	//		for(int i=0;i<_detect_rect_squence[T].detect_rect.size();i++)
//	//		{
//	//			rectangle(m_frame,_detect_rect_squence[T].detect_rect[i],Scalar(255,0,0));
//	//		}
//
//	//		for(int i=0;i<rlt.cols;i++)
//	//		{
//	//			int indx= rlt.at<float>(i);
//	//			rectangle(m_frame,_detect_rect_squence[T+1].detect_rect[indx],Scalar(0,255,0));
//	//		}
//	//		imshow("m_frame",m_frame);
//
//	//	}
//	//	//序列进行前移
//	//	_distance.erase(_distance.begin());
//	//	Mat temp_distance=Mat();
//	//	_distance.push_back(temp_distance);
//	//	_b.erase(_b.begin());
//	//	N_tp1 = _detect_rect_squence[m_nHistoryLen-1].detect_rect.size();
//	//	Mat tempmat=Mat(1,N_tp1,CV_32FC1);
//	//	tempmat.setTo(-1);
//	//	_b.push_back(tempmat);
//	//}
//	
//	return 1;
//}

int CTracker::GetXYChain(vector<DETECTRECT> &_detect_rect_squence,vector<Mat> &_b,int _frame,int _ind,Mat &_xy)
{
	int length= _b.size();
	int tfirst = _frame;
	Mat xy = Mat(7,length,CV_32FC1);
	xy.setTo(0);
	while(_frame<length-1 && _ind>=0)
	{
		Mat tempXYt=xy.colRange(_frame,_frame+1);
		tempXYt.at<float>(0,0)=(float)_detect_rect_squence[_frame].detect_rect_center[_ind].x;
		tempXYt.at<float>(1,0)=(float)_detect_rect_squence[_frame].detect_rect_center[_ind].y;
		
		tempXYt.at<float>(2,0)=(float)_detect_rect_squence[_frame].detect_rect[_ind].x;
		tempXYt.at<float>(3,0)=(float)_detect_rect_squence[_frame].detect_rect[_ind].x;
		tempXYt.at<float>(4,0)=(float)_detect_rect_squence[_frame].detect_rect[_ind].width;
		tempXYt.at<float>(5,0)=(float)_detect_rect_squence[_frame].detect_rect[_ind].height;
		tempXYt.at<float>(6,0)=(float)_detect_rect_squence[_frame].idx[_ind];
		
		if(_ind >= _b[_frame].cols)
			_ind = 0;
		int indnext =(int) _b[_frame].at<float>(0,_ind);
		_b[_frame].at<float>(0,_ind) = -1;
		_ind = indnext;

		_frame++;
	}
	_xy = xy.colRange(tfirst,_frame);
	return 1;
}

int CTracker::LinkDetectionTracklets(vector<DETECTRECT> &_detect_rect_squence,vector<Mat> _b,vector<Mat> _distance,vector<I_TRACK_LINK> &_itl)
{
	cout<<"*********************linkDetectionTracklets*******************"<<endl;
	_itl.clear();
	int length= _detect_rect_squence.size();
	int n=1;
	int cols;
	int rows;
	Mat xy=Mat();
	vector<Mat> tempvector;

	for (int i=0;i<_b.size();i++)
	{
		Mat temp_mat;
		_b[i].copyTo(temp_mat);
		tempvector.push_back(temp_mat);
		//if(m_frame_num<95 && m_frame_num>71)
		//	cout<<temp_mat<<",";
	}
	
	for(int i=0;i<_b.size();i++)
	{
		if(tempvector[i].cols == 0 || tempvector[i].rows==0)
			continue;
		cols =tempvector[i].cols;
		for(int k=0;k<cols;k++)
		{
			if(tempvector[i].at<float>(0,k)>-1)
			{
				GetXYChain(_detect_rect_squence,tempvector,i,k,xy);
				I_TRACK_LINK tempITL;
				int l=xy.cols;

				tempITL.t_start = i + _detect_rect_squence[0].frame_num;
				tempITL.t_end = tempITL.t_start +l-1;
				tempITL.length=l;
				tempITL.omega=Mat(1,l,CV_32FC1);
				tempITL.omega.setTo(1);
				//xy:前两行中心点坐标，后四行为rect区域
				Mat temp_xy_data = xy.rowRange(0,2);
				Mat temp_rect_data = xy.rowRange(2,6);
				temp_xy_data.copyTo(tempITL.xy_data);
				temp_rect_data.copyTo(tempITL.rect_data);
				_itl.push_back(tempITL);
			}
		}
	}
	//if(m_frame_num >=92 && m_frame_num <= 96)
	//{
	//	cout<<"**********xy_data**************"<<endl;
	//	for(int i=0;i<_itl.size();i++)
	//	{
	//		cout<<_itl[i].xy_data<<endl;
	//	}
	//	cout<<"**********xy_data**************"<<endl;
	//}

	tempvector.clear();
	return 1;
}

int growitl(vector<I_TRACK_LINK> _itl,int max_D)
{
	int N_itl = _itl.size();
	BOOL loop_done=FALSE;

	int gap=0;
	while (!loop_done)
	{
		for(int i=0;i<N_itl;i++)
		{
			for(int j=0;j<N_itl;j++)
			{
				gap = (_itl[j].t_start - _itl[i].t_end)-1;
				if(gap==0)
				{
					if(_itl[i].length == 1 && _itl[j].length >1)
					{

					}
				}
			}
		}
	}

	return 1;
}

int CTracker::Get_ITL_Horizon(vector<I_TRACK_LINK> _itl,int _t_start,int _t_end,vector<I_TRACK_LINK> &_itlh)
{
	int N=_itl.size();

	int hormin = _t_start;
	int hormax = _t_end;
	vector<BOOL> f(N);

	for(int i=0;i<N;i++)
	{
		f[i] = !(_itl[i].t_start >= hormax || _itl[i].t_end <= hormin);
		if(f[i] == 1)
		{
			_itlh.push_back(_itl[i]);
			_itlh[i].id = i;
		}
	}

	for(int i=0;i<_itlh.size();i++)
	{
		int si = max(hormin - _itlh[i].t_start,0);
		int ei = max(_itlh[i].t_end - hormax,0);

		int cols = _itlh[i].xy_data.cols;
		Mat tempData= _itlh[i].xy_data.colRange(si, cols-ei);
		tempData.copyTo(_itlh[i].xy_data);
		cout<<_itlh[i].xy_data<<endl;
		cols = _itlh[i].omega.cols;
		Mat tempOmega=_itlh[i].omega.colRange(si,cols-ei);
		tempOmega.copyTo(_itlh[i].omega);
		cout<<_itlh[i].omega<<endl;
		_itlh[i].t_start = max(_itlh[i].t_start,hormin);
		_itlh[i].t_end = min(_itlh[i].t_end ,hormax);

		_itlh[i].length = _itlh[i].t_end - _itlh[i].t_start + 1;
	}

	return 1;
}

//int l2_fastalm_mo(I_TRACK_LINK _itl,RESULTS _p)
//{
//	int D = _itl.xy_data.rows;
//	int N =_itl.xy_data.cols;
//	Mat Omega  = Mat();
//	int nr = ceil((double)N/(D+1))*D;
//	int nc = N - ceil((double)N/(D+1))+1;
//
//	int defMaxIter = 1000;
//	int defTol = 0.0000001;
//
//	if(_itl.omega.empty())
//		Omega = Mat::ones(1,N,CV_8U);
//	else
//		_itl.omega.copyTo(Omega);
//
//	if(_p.nr ==INF)
//		_p.nr = nr;
//	else
//		nr =  _p.nr;
//
//	if(_p.nc ==INF)
//		_p.nc = nc;
//	else
//		nc =  _p.nc;
//
//	if(_p.nc ==INF)
//		_p.nc = nc;
//	else
//		nc =  _p.nc;
//
//	return 1;
//}

int  hankel_mo(Mat _itl_xy,int _nr,int  _nc,Mat &_D,Mat &_H)
{
	int dim = _itl_xy.rows;
	int N = _itl_xy.cols;

	if(_nr ==0)
		_nr = (N-_nc+1)*dim;

	if(_nc ==0)
	{
		_nr = _nc;
		_nc = N - _nr/dim + 1; 

		if((int)_nr%dim!=0)
			printf("error\n");
	}

	int nb= (int)_nr/dim;
	int l = MIN(nb,(int)_nc);
	int D_length = l-1 + N-2*l+2 + l-1; 

	_D = Mat(1,D_length,CV_8U);
	_D.setTo(l);
	for(int i=0;i<l-1;i++)
	{
		_D.data[i] = i + 1;
		_D.data[l-1 + N-2*l+2 + i ] = l-1-i;
	}

	Mat cidx = Mat(1,_nc,CV_8UC1);
	for(int i=0;i<cidx.cols;i++)
	{
		cidx.data[i]=i;
	}
	Mat tempCidx=cv::repeat(cidx,_nr,1);
	cout<<"tempCidx"<<tempCidx<<endl;
	Mat ridx = Mat(1, _nr , CV_8UC1);
	for(int i=0;i<ridx.cols;i++)
	{
		ridx.data[i]=i+1;
	}
	ridx = ridx.t();
	Mat tempRidx=cv::repeat(ridx,1,_nc);
	addWeighted(tempRidx,1,tempCidx,dim,0,_H);
	_H = _H - 1;
	cout<<_H<<endl;
	//注意变换顺序，MATLAB和OPENCV的转换不同
	Mat tempT = Mat();
	tempT=_itl_xy.t();
	tempT=tempT.reshape(0,1);
	cout<<tempT<<endl;

	Mat tempSubs=Mat();
	tempSubs = _H.t();
	tempSubs = tempSubs.reshape(1,1);
	//tempSubs = tempSubs -1;
	tempSubs.convertTo(tempSubs,CV_8U);
	cout<<tempSubs<<endl;
	_H = Mat::zeros(_nc,_nr,CV_32FC1);

	for(int i=0;i<_nr*_nc;i++)
	{
		int idx = tempSubs.data[i];
		_H.at<float>(i) = tempT.at<float>(idx);
	}
	_H=_H.t();
	cout<<_H<<endl;
	return 1;
}

int smot_rank_admm(I_TRACK_LINK _itl,int _eta,RESULTS _p)
{
	int D = _itl.xy_data.rows;
	int N = _itl.xy_data.cols;

	int nr = ceil((double)N/(D+1))*D;
	int nc = N - ceil((double)N/(D+1))+1;

	int defMaxRank = MIN(nr,nc);
	int defMinRank = 1;
	double defLambda  = 0.1;

	int R_max =0;
	if(_p.max_rank == INF)
		R_max = defMaxRank;
	else
		R_max = _p.max_rank;
	R_max = MIN(R_max,nr);
	R_max = MIN(R_max,nc);

	int R_min = 0;
	if(_p.min_rank == INF)
		R_min = defMinRank;
	else
		R_min = _p.min_rank;

	double Lambda=0;
	if(_p.lambda != defLambda)
		Lambda = _p.lambda;
	else
		Lambda = defLambda;

	Mat omega=Mat();
	if (_itl.omega.empty())
	{
		omega = Mat::ones(1,N,CV_32F);
	}
	else
	{
		_itl.omega.copyTo(omega);
	}

	int nCount_total_omega = omega.total();
	int nCount_zero_omega=0;
	for(int i=0;i<nCount_total_omega;i++)
	{
		if(omega.at<float>(i)==0)
			nCount_zero_omega++;
	}

	//if( nCount_zero_omega > 0)
	//	l2_fastalm_mo(_itl,Lambda);
	Mat matH=Mat();
	Mat matD=Mat();
	cout<<"++++++++++++++++++++++"<<endl;
	cout<<_itl.xy_data<<endl;
	hankel_mo(_itl.xy_data,0,R_max,matD,matH);

	SVD matH_SVD(matH);
	cout<<matH_SVD.w<<endl;
	int nCount_matH_SVDW=matH_SVD.w.total();
	int total_matH_SVDW=0;
	for(int i =0;i<nCount_matH_SVDW;i++)
	{
		if(matH_SVD.w.at<float>(i)>_eta)
		{
			total_matH_SVDW ++;
		}

	}
	double R = MAX(R_min,total_matH_SVDW);
	R= MIN(R_max,R);
	return (int)R;
}

int smot_similarity(I_TRACK_LINK _itl_xy1,I_TRACK_LINK _itl_xy2,int _eta,int _gap,RESULTS &_p1,RESULTS &_p2,double& _rank12,double& _s)
{
	int D1 = _itl_xy1.xy_data.rows;
	int T1 = _itl_xy1.xy_data.cols;

	int D2 = _itl_xy2.xy_data.rows;
	int T2 = _itl_xy2.xy_data.cols;

	if(D1!=D2)
		printf("Error:Input dimensions do not agree.\n");

	double defGap      = 0;
	Mat Omega1   = Mat();
	Mat Omega2   = Mat();
	double defRank1    = INF;
	double defRank2    = INF;
	BOOL defQCheck   = false;

	double gap = 0;
	if(_gap == 0)
		gap = defGap;
	else
		gap = _gap;

	if(_itl_xy1.omega.empty())
		Omega1   = Mat::ones(1,T1,CV_8U);
	else
		_itl_xy1.omega.copyTo(Omega1);

	if(_itl_xy2.omega.empty())
		Omega2   = Mat::ones(1,T1,CV_8U);
	else
		_itl_xy2.omega.copyTo(Omega2);

	double rank1 = 0;
	if(_p1.rank == INF)
		rank1 = defRank1;
	else
		rank1 = _p1.rank;

	double rank2 = 0;
	if(_p2.rank == INF)
		rank2 = defRank2;
	else
		rank2 = _p2.rank;

	cout<<Omega1<<endl;

	//修改全局变量
	BOOL qcheck=_p1.qcheck;

	_p1.rank =(double)smot_rank_admm(_itl_xy1,_eta,_p1);
	_p2.rank =(double)smot_rank_admm(_itl_xy2,_eta,_p2);
	rank1 = _p1.rank ;
	rank2 = _p2.rank ;
	if(qcheck)
	{
		int nr= (int)MIN(T1-rank1,T2-rank2);
		nr= nr/D1*D1;
		Mat H1=Mat();
		Mat H2=Mat();
		Mat tempD;
		hankel_mo(_itl_xy1.xy_data,nr,0,tempD,H1);
		hankel_mo(_itl_xy1.xy_data,nr,0,tempD,H2);
		Mat combineMat;
		combineMat.push_back(H1);
		combineMat.push_back(H2);
		SVD combineMat_svd(combineMat);
		int nCount=combineMat_svd.w.total();
		double  sum=0;
		for(int i=0;i<nCount;i++)
		{
			if(combineMat_svd.w.data[i]>_eta)
				sum =sum  +combineMat_svd.w.data[i];
		}
		_rank12 = sum;

		if(_rank12 > rank1+rank2)
		{
			_s =-INF;
			return 1;
		}
	}

	Mat tempMat=Mat();

	Mat XY12_data=Mat();
	tempMat= _itl_xy1.xy_data.t();
	XY12_data.push_back(tempMat);
	tempMat= Mat::zeros((int)gap,(int)D1,CV_32F);
	XY12_data.push_back(tempMat);
	tempMat= _itl_xy2.xy_data.t();
	XY12_data.push_back(tempMat);
	XY12_data = XY12_data.t();

	Mat Omega12 = Mat();
	tempMat = Omega1.t();
	Omega12.push_back(tempMat);
	tempMat= Mat::zeros((int)gap,1,CV_32F);
	Omega12.push_back(tempMat);
	tempMat = Omega2.t();
	Omega12.push_back(tempMat);
	Omega12=Omega12.t();

	I_TRACK_LINK itl_xy12;
	XY12_data.copyTo(itl_xy12.xy_data);
	Omega12.copyTo(itl_xy12.omega);
	RESULTS p;
	p.min_rank = MIN(rank1,rank2);
	p.max_rank = rank1+rank2;
	_rank12=smot_rank_admm(itl_xy12,_eta,p);

	_s=(rank1+rank2)/_rank12 -1;
	if(_s <0.000005)
		_s=-INF;

	//binlong ：to be continue
	return 1;
}

void similarity_itl(I_TRACK_LINK& _itl_i,I_TRACK_LINK& _itl_j,DEFAULT_PARAMS _params,double& _rank12,double& _s)
{
	double defMaxHorizon = INF;
	double defMaxGap = INF;
	double defMaxSlope = INF;

	double hor_max=0;
	if(_params.hor_max==INF)
		hor_max = defMaxHorizon;
	else
		hor_max = _params.hor_max;

	double gap_max=0;
	if(_params.gap_max==INF)
		gap_max = defMaxGap;
	else
		gap_max = _params.gap_max;

	double slope_max=0;
	if(_params.slope_max==INF)
		slope_max = defMaxSlope;
	else
		slope_max = _params.slope_max;

	// Individual Ranks
	double rank1 = _itl_i.rank;
	double rank2 = _itl_j.rank;
	double rank12 =INF;

	if(_itl_i.length<=2 && _itl_j.length<=2)
		return ;

	_params.gap = _itl_j.t_start - _itl_i.t_end -1;
	double slope = 0;
	Mat tempMat=Mat();
	addWeighted(_itl_j.xy_data.col(0),1,_itl_i.xy_data.col(_itl_i.xy_data.cols-1),-1,0,tempMat);
	slope=norm(tempMat,4)/(_params.gap+1);


	RESULTS _itl_p1;
	RESULTS _itl_p2;
	if( _params.gap >= 0 && _params.gap < _params.gap_max && slope < (_params.slope_max*2))
		smot_similarity(_itl_i,_itl_j,_params.eta_max,_params.gap,_itl_p1,_itl_p2,_rank12,_s);
	else if (0 & (_itl_j.t_start > _itl_i.t_start ) && (_itl_j.t_end < _itl_i.t_end))
	{
		int i=0;
		int rt_start = _itl_j.t_start - _itl_i.t_start + 1;
		int rt_end = rt_start + _itl_j.length - 1;
		int total_sum=0;
		for(int i=0;i<rt_end - rt_start;i++)
		{
			if(_itl_i.omega.at<float>(i) ==0)
				total_sum++;
		}
		int rmax =0;
		if(total_sum == _itl_j.length)
			rmax = rank1 + rank2;

		/*	
		
		if sum(itl1.omega(rt_start:rt_end)==0) == itl2.length
		% compute the joint rank
		rmax = r1+r2;

		% TODO: Rewrite the following
		[s r1 r2 r12] = smot_similarity([itl1.data(:,1:rt_start-1) itl2.data itl1.data(:,rt_end+1:end)],eta_max,...
		'gap',gap,...
		'rank1',itl1.rank,'rank2',itl2.rank,...
		'omega1',itl1.omega,'omega2',itl2.omega);

		%'omega',[itl1.omega(1:rt_start-1) itl2.omega itl1.omega(rt_end+1:end)]);
		

		end*/
	}


}

int CTracker::Compute_ITL_Similarity_Matrix(vector<I_TRACK_LINK> &_itl,DEFAULT_PARAMS _param)
{
	int N=_itl.size();

	double max_gap=0;
	int nCount=0;
	for(int i=0;i<N;i++)
	{
		if(_itl[i].length>1)
		{
			max_gap = max_gap + _itl[i].length;
			nCount++;
		}
	}

	max_gap = max_gap / nCount / 2;

	double max_slope = 0;
	cout<<"********************************************"<<endl;
	int cols = 0;
	for(int i=0;i<N;i++)
	{
		cols = _itl[i].xy_data.cols;
		Mat tempData1=_itl[i].xy_data.colRange(0,cols-1);
		Mat tempData2=_itl[i].xy_data.colRange(1,cols);

		Mat dx;
		cv::absdiff(tempData1,tempData2,dx);
		Mat norm_dx = dx.mul(dx);
		norm_dx.convertTo(norm_dx,CV_32F);
		Mat sum = Mat::zeros(1,norm_dx.cols,CV_32F);
		for (int i=0;i<norm_dx.rows;i++)
		{
			add(norm_dx.row(i),sum,sum);
		}

		sqrt(sum,norm_dx);
		cout<<norm_dx<<endl;
		Mat maxMat;
		max(norm_dx,max_slope,maxMat);
		cout<<"maxMat"<<maxMat<<endl;
		minMaxLoc(maxMat,NULL,&max_slope,NULL,NULL);
	}

	Mat matS = Mat::ones(N,N,CV_32FC1);
	matS.setTo(NINF);


	_param.slope_max = max_slope;
	_param.gap_max = max_gap;

	double rank12=INF;
	double s=0;
	for(int i=0;i<N;i=i++)
	{
		for(int j=0;j<N;j++)
		{
			if( i == j)
				s = NINF;
			else
			{
				similarity_itl(_itl[i],_itl[j],_param,rank12,s);
			}
			matS.at<float>(i,j)= (float)s;
		}
	}
	cout<<matS<<endl;
	return 1;
}

int  CTracker::Associate_ITL(vector<I_TRACK_LINK> _itl,int _t_start,int _t_end)
{
	int N=_itl.size();
	vector<I_TRACK_LINK> itlh;

	//去除过短的跟踪线
	int N_itlh = itlh.size();
	int i=0;
	while(i<N_itlh)
	{
		if(itlh[i].length<=2)
		{
			itlh.erase(itlh.begin()+i);
			N_itlh--;
			continue;
		}
		i++;
	}
	DEFAULT_PARAMS params;

	int Nnew =0;
	int dN=1;
	if(!itlh.empty())
	{
		while(dN>0)
		{
			Compute_ITL_Similarity_Matrix(itlh,params);
			Nnew = itlh.size();
			dN = N - Nnew;
			N = Nnew;
		}
	}


	return 1;
}

int CTracker::Compute_DetectionTracklets_Similarity(vector<I_TRACK_LINK> &_itl)
{
	//无可跟踪目标
	if(m_itl.size()<=0 && _itl.size()<=0)
	{
		m_itl.clear();
	}
	//出现新的跟踪目标
	else if(m_itl.size()<=0 && _itl.size()>0)
	{
		m_itl = _itl;
		for(int i=0;i<m_itl.size();i++)
		{
			m_itl[i].id=GetTickCount();
		}	
	}
	else
	{
		//检测是否是现有队列中的跟踪目标
		for(int i=0;i<_itl.size();i++)
		{
			for(int j=0;j< m_itl.size();j++)
			{
				//判断时间是否重复
			/*	if(m_itl[j].t_end == _itl[i].t_end || )*/

			}
		}
		

	}
	
	return 1;
}

int CTracker::LinkDetectionTracklets1(vector<DETECTRECT> &_detect_rect_squence,vector<Mat> _b,vector<Mat> _distance,vector<I_TRACK_LINK> &_itl)
{
	int b_length= _b.size();
	int T =b_length-1;
	int Tnp1=T-1;
	if(b_length < 3)
		return -1;
	//当前T时刻,b_length-2
	int T_b_cols = _b[T].cols;
	int T_b_rows = _b[T].rows;

	//T-1时刻,b_length-1
	int Tnp1_b_cols = _b[Tnp1].cols;
	int Tnp1_b_rows = _b[Tnp1].rows;
	//没有跟踪目标
	if(T_b_cols == 0 || T_b_rows==0)
		return -1;
	
	
	//发现跟踪目标，查找与T-1时刻的关系
	//T-1为空，新发现目标
	for(int i=0;i<T_b_cols;i++)
	{
		int object_id=0;
		int ind=-1;
		if(m_frame_num >=91)
			int ll=0;
		//上一时刻无跟踪目标
	/*	if(Tnp1_b_cols ==0)
		{
			object_id = GetTickCount();
			_detect_rect_squence[T].object_id[i]=object_id;
			Mat tempXYt = Mat(7,1,CV_32FC1);
			tempXYt.at<float>(0,0)=(float)_detect_rect_squence[T].detect_rect_center[i].x;
			tempXYt.at<float>(1,0)=(float)_detect_rect_squence[T].detect_rect_center[i].y;

			tempXYt.at<float>(2,0)=(float)_detect_rect_squence[T].detect_rect[i].x;
			tempXYt.at<float>(3,0)=(float)_detect_rect_squence[T].detect_rect[i].x;
			tempXYt.at<float>(4,0)=(float)_detect_rect_squence[T].detect_rect[i].width;
			tempXYt.at<float>(5,0)=(float)_detect_rect_squence[T].detect_rect[i].height;
			tempXYt.at<float>(6,0)=(float)object_id;

			Mat temp_xy_data=tempXYt.rowRange(0,2);
			Mat temp_rect_data=tempXYt.rowRange(2,6);
			Mat temp_rect_id = tempXYt.rowRange(6,7);
			
			I_TRACK_LINK new_itl;
			new_itl.id = object_id;
			new_itl.t_start = _detect_rect_squence[b_length-2].frame_num;
			new_itl.t_end = new_itl.t_start;
			new_itl.length = 1;
			temp_xy_data.copyTo(new_itl.xy_data);
			_itl.push_back(new_itl);
		}
		else
		{*/
			//匹配有无此跟踪目标
		for(int j=0;j<Tnp1_b_cols;j++)
		{
			if(_b[Tnp1].at<float>(0,j) == i )
			{
				ind = j;
			}
		}
		if(ind==-1)
		{
			object_id = GetTickCount();
			_detect_rect_squence[T].object_id[i]=object_id;
			Mat tempXYt = Mat(7,1,CV_32FC1);
			tempXYt.at<float>(0,0)=(float)_detect_rect_squence[T].detect_rect_center[i].x;
			tempXYt.at<float>(1,0)=(float)_detect_rect_squence[T].detect_rect_center[i].y;

			tempXYt.at<float>(2,0)=(float)_detect_rect_squence[T].detect_rect[i].x;
			tempXYt.at<float>(3,0)=(float)_detect_rect_squence[T].detect_rect[i].x;
			tempXYt.at<float>(4,0)=(float)_detect_rect_squence[T].detect_rect[i].width;
			tempXYt.at<float>(5,0)=(float)_detect_rect_squence[T].detect_rect[i].height;
			tempXYt.at<float>(6,0)=(float)object_id;

			Mat temp_xy_data=tempXYt.rowRange(0,2);
			Mat temp_rect_data=tempXYt.rowRange(2,6);
			Mat temp_rect_id = tempXYt.rowRange(6,7);

			I_TRACK_LINK new_itl;
			new_itl.id = object_id;
			new_itl.t_start = _detect_rect_squence[T].frame_num;
			new_itl.t_end = new_itl.t_start;
			new_itl.length = 1;
			temp_xy_data.copyTo(new_itl.xy_data);
			_itl.push_back(new_itl);
		}
		else
		{
			if(_detect_rect_squence[Tnp1].detect_rect.size() >  _detect_rect_squence[T].detect_rect.size() )
			{
				Mat rect_distance;
				Mat rlt;
				CalRectDistance(_detect_rect_squence[T],_detect_rect_squence[Tnp1],m_ratio_threhold,rect_distance,rlt);
				cout<<rlt<<endl;
				cout<<rect_distance<<endl;
				for(int k=0;k<rlt.cols;k++)
				{
					if(rlt.at<float>(0,k) == i )
					{
						ind = k;
					}
				}
			}
			else
			{
				object_id = _detect_rect_squence[Tnp1].object_id[ind];
				_detect_rect_squence[T].object_id[i] = _detect_rect_squence[Tnp1].object_id[ind];
				for(int j=0;j<_itl.size();j++)
				{
					if(object_id == _itl[j].id)
					{
						_itl[j].length ++;
						_itl[j].t_end = _detect_rect_squence[T].frame_num;

						Mat tempXYt = Mat(7,1,CV_32FC1);
						tempXYt.at<float>(0,0)=(float)_detect_rect_squence[T].detect_rect_center[i].x;
						tempXYt.at<float>(1,0)=(float)_detect_rect_squence[T].detect_rect_center[i].y;

						tempXYt.at<float>(2,0)=(float)_detect_rect_squence[T].detect_rect[i].x;
						tempXYt.at<float>(3,0)=(float)_detect_rect_squence[T].detect_rect[i].x;
						tempXYt.at<float>(4,0)=(float)_detect_rect_squence[T].detect_rect[i].width;
						tempXYt.at<float>(5,0)=(float)_detect_rect_squence[T].detect_rect[i].height;
						tempXYt.at<float>(6,0)=(float)object_id;

						Mat temp_xy_data=tempXYt.rowRange(0,2);
						Mat temp_rect_data=tempXYt.rowRange(2,6);
						Mat temp_rect_id = tempXYt.rowRange(6,7);

						Mat temp_meger=Mat(2,_itl[j].length,CV_32FC1);
						_itl[j].xy_data.copyTo(temp_meger.colRange(0,_itl[j].length-1));
						temp_xy_data.copyTo(temp_meger.colRange(_itl[j].length-1,_itl[j].length));
						temp_meger.copyTo(_itl[j].xy_data);

					}
				}
			}
		}
	//	}
	}
	return 1;
}