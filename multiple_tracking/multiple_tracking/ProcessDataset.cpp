#include "function.h"
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
	
	//求取最大序列长度
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
		//检查每个序列
		for(int k=0;k<N;k++)
		{
			if(i>=_dataset[k].t_start && i<=_dataset[k].t_end)
			{
				if(_dataset[k].omega[i-_dataset[k].t_start])
				{
					_detect_rect[i].idx.push_back(k);
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


void DiffMat(Mat _a,Mat &_b)  //求向量B的一阶差分 功能等价matlab里的diff
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
int frame=0;
void nonunique(Mat _a,Mat &_b)
{
	
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

	cout<<"**************************"<<endl<<sorted<<endl;
	cout<<tempB<<endl;

	if(nNonZero ==1 && numelA >1)
		tempB.copyTo(_b);
	else
	{

		if(frame==71)
			int kkkl=0;
		vector<int> nhistCount(nNonZero);
		
		for(int k=0;k<numelA;k++)
		{
			for(int i=0;i<nNonZero;i++)
			{
				if(tempB.at<float>(0,i) == tempA.at<float>(0,k))
					nhistCount[i]++;
			}
		}
		for(int k=0;k<nNonZero;k++)
		{
			if(nhistCount[k]>1)
			{

			}
		}
		/*cout<<nhistCount<<endl;*/
	}
	
}

int findAssociations(vector<DETECTRECT> &_detect_rect,int _ratio_threhold = 3 )
{
	int length= _detect_rect.size();
	vector<Mat> B;
	vector<Mat> distance;
	Mat ratio;
	Mat f;
	double lastminD2=0;
	for(int T=0;T<length-1;T++)
	{
		frame++;
		cout<<frame<<endl;
		//t时刻的检测框数量
		int N_t=_detect_rect[T].detect_rect.size();
		//t+1时刻的检测框数量
		int N_tp1=_detect_rect[T+1].detect_rect.size();

		if( N_t > 0 && N_tp1> 0)
		{
			Mat mat_distance = Mat(N_t,N_tp1,CV_32FC1);
			//计算检测框间的距离
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
			distance.push_back(mat_distance);
			//检测前两位距离比值
			Mat Dsorted;
			cv::sort(mat_distance,Dsorted,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING );
			Mat DIdx;
			cv::sortIdx(mat_distance,DIdx,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING );

			if(N_tp1>1)
			{
				Mat A=Dsorted.colRange(0,1);
				Mat B=Dsorted.colRange(1,2);
				Mat C;
				divide(B,A,ratio);
				f=ratio>_ratio_threhold;
					
				minMaxLoc(Dsorted,NULL,&lastminD2,NULL,NULL);    // 不需要的置为0 
			}
			else if(lastminD2>0)
			{
				Mat A=Dsorted.colRange(0,1);
				divide(lastminD2,A,ratio);
				f=ratio>_ratio_threhold;
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
			B.push_back(tempmat);

			Mat rlt;
			nonunique(tempmat,rlt);
			//检测重复的idx
	/*		nonunique(tempmat);*/
		}	
	}

	return 1;
}





