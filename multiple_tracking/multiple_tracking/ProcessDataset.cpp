#include "function.h"
int frame=0;
int read_dataset(char* _file_path,vector<DATASET> &_dataset)
{
	FILE* fp;
	fp= fopen(_file_path,"rb");
	int N=0;
	fscanf(fp,"%d",&N);
	_dataset.resize(N);
	for(int i=0;i<N;i++)
	{
		fscanf(fp,"%d %d %d",&_dataset[i].id,&_dataset[i].t_start,&_dataset[i].t_end);
		_dataset[i].t_start = _dataset[i].t_start -1;
		_dataset[i].t_end = _dataset[i].t_end -1;
		_dataset[i].length = _dataset[i].t_end - _dataset[i].t_start+1;
		_dataset[i].omega.resize(_dataset[i].length);
		_dataset[i].rect.resize(_dataset[i].length);
		for(int k=0;k<_dataset[i].length;k++)
		{
			fscanf(fp,"%f",&_dataset[i].rect[k].x);
		}

		for(int k=0;k<_dataset[i].length;k++)
		{
			fscanf(fp,"%f",&_dataset[i].rect[k].y);
		}

		for(int k=0;k<_dataset[i].length;k++)
		{
			fscanf(fp,"%f",&_dataset[i].rect[k].width);
		}

		for(int k=0;k<_dataset[i].length;k++)
		{
			fscanf(fp,"%f",&_dataset[i].rect[k].height);
		}

		for(int k=0;k<_dataset[i].length;k++)
		{
			fscanf(fp,"%d",&_dataset[i].omega[k]);
			_dataset[i].rect[k].center_point[0] =_dataset[i].rect[k].x + _dataset[i].rect[k].width/2;
			_dataset[i].rect[k].center_point[1] =_dataset[i].rect[k].y + _dataset[i].rect[k].height/2;
		}
	}
	return 1;
}

int ProcessDataSet(vector<DETECTRECT> &_detect_rect,vector<DATASET> &_dataset)
{
	int N=_dataset.size();
	int T_start=_dataset[0].t_start;
	int T_end=_dataset[0].t_end;
	
	//��ȡ������г���
	for(int i=0;i<N;i++)
	{
		if(_dataset[i].t_end>T_end)
			T_end = _dataset[i].t_end;
		if(T_start>_dataset[i].t_start)
			T_start=_dataset[i].t_start;
	}
	
	int length= T_end -T_start +1;
	_detect_rect.resize(length);
	for(int i=T_start;i < length;i++)
	{
		//���ÿ������
		for(int k=0;k<N;k++)
		{
			if(i>=_dataset[k].t_start && i<=_dataset[k].t_end)
			{
				if(_dataset[k].omega[i-_dataset[k].t_start])
				{
					_detect_rect[i].idx.push_back(_dataset[k].id);
					_detect_rect[i].detect_rect.push_back(_dataset[k].rect[i-_dataset[k].t_start]);
				}
				else
				{
					continue;
				}
			}
			else
			{
				continue;
			}
		}
	}
	return 1;
}

int initial_track(char* _dataset_name)
{
	return 1;
}

void DiffMat(Mat _a,Mat &_b)  //������B��һ�ײ�� ���ܵȼ�matlab���diff
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

void nonunique(Mat _a,Mat &_b)
{
	if(_a.empty())
		return;
	int rows = _a.rows;
	int cols = _a.cols;

	int rowvec = (rows == 1) &&(cols>1);
	//����Ԫ������
	int numelA = cols * rows;
	if(numelA == 1)
	{
		_a.copyTo(_b);
		_b.convertTo(_b,CV_32F);
		_b = _b-1;
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

	Mat d = Mat(1,cols*rows,CV_8U);
	Mat tempd= (db!=0);
	int tempd_size=tempd.channels()*tempd.cols*tempd.step*tempd.elemSize();
	memcpy(d.data,tempd.data,tempd_size);
	d.at<uchar>(0,cols*rows-1)=255;
	
	int nNonZero=countNonZero(d);
	
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
		vector<int> nhistCount(nNonZero);
		
		for(int k=0;k<numelA;k++)
		{
			for(int i=0;i<nNonZero;i++)
			{
				if(tempB.at<float>(0,i) == tempA.at<float>(0,k))
					nhistCount[i]++;
			}
		}
		
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
					for(int j=0;j<numelA;j++)
					{
						if(tempA.at<float>(0,j) == tempVectorB[k] )
							tempA.at<float>(0,j) = 0;
					}
				}
			}
		}
		tempA.copyTo(_b);
	}
	_b = _b - 1;
}

<<<<<<< .mine

int findAssociations(vector<DETECTRECT> &_detect_rect,int _ratio_threhold,vector<Mat> &_b,vector<Mat> &_distance)
=======
int frame=0;
int findAssociations(vector<DETECTRECT> &_detect_rect,int _ratio_threhold = 3 )
>>>>>>> .r16
{
	int length= _detect_rect.size();

	Mat ratio;
	Mat f;
	double lastminD2=0;
	for(int T=0;T<length-1;T++)
	{
		cout<<"**************************"<<endl;
		frame++;
		cout<<"frame="<<frame<<endl;
		//tʱ�̵ļ�������
		int N_t=_detect_rect[T].detect_rect.size();
		//t+1ʱ�̵ļ�������
		int N_tp1=_detect_rect[T+1].detect_rect.size();

		if( N_t > 0 && N_tp1> 0)
		{
			Mat mat_distance = Mat(N_t,N_tp1,CV_32FC1);
			//��������ľ���
			for(int NT_i=0 ; NT_i< N_t ;NT_i++)
			{
				for(int NT_ip1=0;NT_ip1<N_tp1;NT_ip1++)
				{
					mat_distance.at<float>(NT_i,NT_ip1)=\
						pow((_detect_rect[T].detect_rect[NT_i].center_point[0]-_detect_rect[T+1].detect_rect[NT_ip1].center_point[0]),2)\
						+pow((_detect_rect[T].detect_rect[NT_i].center_point[1]-_detect_rect[T+1].detect_rect[NT_ip1].center_point[1]),2);
					mat_distance.at<float>(NT_i,NT_ip1) = sqrt(mat_distance.at<float>(NT_i,NT_ip1));
				}
			}
			_distance.push_back(mat_distance);
			
			
			//���ǰ��λ�����ֵ
			Mat Dsorted;
			cv::sort(mat_distance,Dsorted,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING );
			Mat DIdx;
			cv::sortIdx(mat_distance,DIdx,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING );
			DIdx = DIdx+1;

			if(N_tp1>1)
			{
				Mat A=Dsorted.colRange(0,1);
				Mat B=Dsorted.colRange(1,2);
				Mat C;
				divide(B,A,ratio);
				f=ratio>_ratio_threhold;
				for(int i=0;i<A.rows;i++)
				{
					if(A.at<float>(i,0)==0)
						f.at<uchar>(i,0) = 255;
				}
				minMaxLoc(Dsorted,NULL,&lastminD2,NULL,NULL);    // ����Ҫ����Ϊ0 
			}
			else if(lastminD2>0)
			{
				Mat A=Dsorted.colRange(0,1);
				divide(lastminD2,A,ratio);
				f=ratio>_ratio_threhold;
				for(int i=0;i<A.rows;i++)
				{
					if(A.at<float>(i,0)==0)
						f.at<uchar>(i,0) = 255;
				}
			}
			else
			{
				Mat A=Dsorted.colRange(0,1);
				ratio = A*0;
				f= ratio>_ratio_threhold;
			}

			Mat A=DIdx.colRange(0,1);
			A.convertTo(A,CV_8UC1);
			Mat tempmat=A.mul(f/255);
			Mat rlt;
<<<<<<< .mine
			cout<<"tempmat="<<tempmat<<endl;
			
			//����ظ���idx
=======
			cout<<"tempmat="<<tempmat<<endl;
			//����ظ���idx
>>>>>>> .r16
			nonunique(tempmat,rlt);
<<<<<<< .mine
			cout<<"rlt=\t"<<rlt<<endl;
			_b.push_back(rlt);
=======
			cout<<"rlt=\t"<<rlt<<endl;
			B.push_back(rlt);
>>>>>>> .r16
		}	
	}

<<<<<<< .mine
	int N_tp1 = _detect_rect[length-1].detect_rect.size();
	Mat tempmat=Mat(1,N_tp1,CV_32FC1);
	tempmat.setTo(-1);
	cout<<tempmat<<endl;
	_b.push_back(tempmat);

=======
	int N_tp1 = _detect_rect[length-1].detect_rect.size();
	Mat tempmat=Mat(1,N_tp1,CV_32FC1);
	tempmat.setTo(0);
	cout<<tempmat<<endl;
	B.push_back(tempmat);

>>>>>>> .r16
	return 1;
}

int getxychain(vector<DETECTRECT> &_detect_rect,vector<Mat> &_b,int _frame,int _ind,Mat &_xy)
{
	int length= _detect_rect.size();
	int tfirst = _frame;
	Mat xy = Mat(2,length,CV_32FC1);
	xy.setTo(0);

	while(_frame<length && _ind>=0)
	{
		Mat tempXYt=xy.colRange(_frame,_frame+1);
		tempXYt.at<float>(0,0)=_detect_rect[_frame].detect_rect[_ind].center_point[0];
		tempXYt.at<float>(1,0)=_detect_rect[_frame].detect_rect[_ind].center_point[1];
		int indnext =(int) _b[_frame].at<float>(0,_ind);
		_b[_frame].at<float>(0,_ind) = -2;
		_ind = indnext;
		_frame++;
	}
	_xy = xy.colRange(tfirst,_frame);
	cout<<_xy<<endl;
	return 1;
}



int linkDetectionTracklets(vector<DETECTRECT> &_detect_rect,vector<Mat> _b,vector<Mat> _distance)
{
	cout<<"*********************linkDetectionTracklets*******************"<<endl;

	int length= _detect_rect.size();
	int n=1;
	int cols;
	int rows;
	Mat xy=Mat();
	for(int i=0;i<length;i++)
	{
		cols =_b[i].cols;
		cout<<"_b["<<i<<"]="<<_b[i]<<endl;
		for(int k=0;k<cols;k++)
		{
			if(_b[i].at<float>(0,k)>-2)
			{
				getxychain(_detect_rect,_b,i,k,xy);

			}
		}
	}

	return 1;
}


int growitl(vector<Mat> &_itl,int max_D)
{

	return 1;
}