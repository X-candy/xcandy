#include "function.h"
#include "Params.h"
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


int findAssociations(vector<DETECTRECT> &_detect_rect,int _ratio_threhold,vector<Mat> &_b,vector<Mat> &_distance)
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
			_distance.push_back(mat_distance);
			
			
			//检测前两位距离比值
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
				minMaxLoc(Dsorted,NULL,&lastminD2,NULL,NULL);    // 不需要的置为0 
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

			//检测重复的idx
			nonunique(tempmat,rlt);
			_b.push_back(rlt);
		}	
	}

	int N_tp1 = _detect_rect[length-1].detect_rect.size();
	Mat tempmat=Mat(1,N_tp1,CV_32FC1);
	tempmat.setTo(-1);

	_b.push_back(tempmat);

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
	return 1;
}

int linkDetectionTracklets(vector<DETECTRECT> &_detect_rect,vector<Mat> _b,vector<Mat> _distance,vector<I_TRACK_LINK> &_itl)
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
		for(int k=0;k<cols;k++)
		{
			if(_b[i].at<float>(0,k)>-2)
			{
				getxychain(_detect_rect,_b,i,k,xy);
				I_TRACK_LINK tempITL;
				int l=xy.cols;
	
				tempITL.t_start = i;
				tempITL.t_end = i+l-1;
				tempITL.length=l;
				tempITL.omega=Mat(1,l,CV_32FC1);
				tempITL.omega.setTo(1);
				xy.copyTo(tempITL.xy_data);
				_itl.push_back(tempITL);
			}
		}
	}
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

int get_itl_horizon(vector<I_TRACK_LINK> _itl,int _t_start,int _t_end,vector<I_TRACK_LINK> &_itlh)
{
	int N=_itl.size();

	int hormin = _t_start;
	int hormax = _t_end;
	vector<BOOL> f(N);

	for(int i=0;i<N;i++)
	{
		f[i] = !(_itl[i].t_start >= hormax || _itl[i].t_start <= hormin);
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
		cols = _itlh[i].omega.cols;
		Mat tempOmega=_itlh[i].omega.colRange(si,cols-ei);
		tempOmega.copyTo(_itlh[i].omega);

		_itlh[i].t_start = max(_itlh[i].t_start,hormin);
		_itlh[i].t_end = min(_itlh[i].t_end ,hormax);

		_itlh[i].length = _itlh[i].t_end - _itlh[i].t_start + 1;
	}

	return 1;
}

int similarity_itl(I_TRACK_LINK _itl_i,I_TRACK_LINK _itl_j,DEFAULT_PARAMS _params)
{
	//to be continue

	return 1;
}

int smot_similarity(vector<I_TRACK_LINK> _itl_xy1,vector<I_TRACK_LINK> _itl_xy2,int _eta)
{
	//to be continue
	return 1;
}

int l2_fastalm_mo(I_TRACK_LINK _itl)
{//to be continue
	return 1;
}

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
	cout<<tempCidx<<endl;

	Mat ridx = Mat(1, _nr , CV_8UC1);
	for(int i=0;i<ridx.cols;i++)
	{
		ridx.data[i]=i+1;
	}
	ridx = ridx.t();
	Mat tempRidx=cv::repeat(ridx,1,_nc);
	addWeighted(tempRidx,1,tempCidx,dim,0,_H);
	_H = _H - 1;

	//注意变换顺序，MATLAB和OPENCV的转换不同
	Mat tempT = Mat();
	tempT=_itl_xy.t();
	tempT=tempT.reshape(0,1);
	cout<<tempT<<endl;

	Mat tempSubs=Mat();
	tempSubs = _H.t();
	tempSubs = tempSubs.reshape(0,1);
	tempSubs = tempSubs -1;
	tempSubs.convertTo(tempSubs,CV_8U);

	_H = Mat::zeros(_nc,_nr,CV_32FC1);
	cout<<_H<<endl;
	for(int i=0;i<_nr*_nc;i++)
	{
		int idx = tempSubs.data[i];
		_H.data[i] = tempT.data[idx];
	}
	_H=_H.t();
	return 1;
}

int smot_rank_admm(I_TRACK_LINK _itl,int _eta)
{
	int D = _itl.xy_data.rows;
	int N = _itl.xy_data.cols;

	int nr = ceil((double)N/(D+1))*D;
	int nc = N - ceil((double)N/(D+1))+1;

	int defMaxRank = MIN(nr,nc);
	int defMinRank = 1;
	Mat defOmega   = Mat::ones(1,N,CV_32F);
	double defLambda  = 0.1;

	int R_max = defMaxRank;
	R_max = MIN(R_max,nr);
	R_max = MIN(R_max,nc);
	int R_min = defMinRank;

	Mat matH=Mat();
	Mat matD=Mat();
	hankel_mo(_itl.xy_data,nr,nc,matD,matH);

	SVD matH_SVD(matH);
	int nCount_matH_SVDW=matH_SVD.w.total();
	double sum_matH_SVDW=0;
	for(int i =0;i<nCount_matH_SVDW;i++)
	{
		if(matH_SVD.w.data[i]>_eta)
		{
			sum_matH_SVDW = matH_SVD.w.data[i] + sum_matH_SVDW;
		}

	}
	R = MAX(R_min,R);
	R= MIN(R_max,R);
	return 1;
}

int compute_itl_similarity_matrix(vector<I_TRACK_LINK> &_itl,DEFAULT_PARAMS _param)
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
		sqrt(norm_dx,norm_dx);
		Mat maxMat;
		max(norm_dx,max_slope,maxMat);
		minMaxLoc(maxMat,NULL,&max_slope,NULL,NULL);
	}

	Mat S = Mat::ones(N,N,CV_32FC1);
	S.setTo(PINF);
	cout<<S<<endl;

	float s=0;
	for(int i=0;i<N;i++)
	{
		for(int j=0;j<N;j++)
		{
			if( i == j)
				s = NINF;
			else
				int kkk=0;
			//to be continue;
		}
	}
	return 1;
}

int associate_itl(vector<I_TRACK_LINK> _itl,int _t_start,int _t_end)
{
	int N=_itl.size();
	vector<I_TRACK_LINK> itlh;
	get_itl_horizon(_itl,_t_start,_t_end,itlh);
	
	//去除过短的跟踪线
	int N_itlh = itlh.size();
	int i=0;
	while(i<N_itlh)
	{
		if(itlh[i].id<=2)
		{
			itlh.erase(itlh.begin()+i);
			N_itlh--;
			continue;
		}
		i++;
	}

	
	if(!itlh.empty())
	{
		int dN=1;

		while(dN>0)
		{

		}

	}
	
	return 1;
}

