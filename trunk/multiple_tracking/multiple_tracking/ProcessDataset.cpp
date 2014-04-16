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
	d.setTo(255);
	Mat tempd= (db!=0);
	int tempd_size=tempd.channels()*tempd.cols*tempd.step*tempd.elemSize();
	for(int i=0;i<tempd.cols*tempd.rows;i++)
	{
		d.at<uchar>(i) = tempd.at<uchar>(i);
	}
	//memcpy(d.data,tempd.data,tempd_size);
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
		if(tfirst ==129 && _ind ==5 )
			cout<<"_frame="<<_frame<<endl;

		if(_frame==339)
			int kkklkl=0;
		Mat tempXYt=xy.colRange(_frame,_frame+1);
		if(_detect_rect[_frame].detect_rect.size()!=0 && _ind<_detect_rect[_frame].detect_rect.size())
		{
			tempXYt.at<float>(0,0)=_detect_rect[_frame].detect_rect[_ind].center_point[0];
			tempXYt.at<float>(1,0)=_detect_rect[_frame].detect_rect[_ind].center_point[1];
			//出错原因，矩阵超限值，公式不应该出错的，暂未保护
			double indnext = _b[_frame].at<float>(0,_ind);
			_b[_frame].at<float>(0,_ind) = -2;
			_ind = indnext;
		}
		
		
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
		cout<<"i="<<i<<endl;
		
		cols =_b[i].cols;
		for(int k=0;k<cols;k++)
		{
			cout<<"k="<<k<<endl;
			if(i==129 && k==5)
				int kkkklkds=0;
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
		f[i] = !(_itl[i].t_start >= hormax || _itl[i].t_end <= hormin);
		cout<<f[i]<<"\t";
		if(f[i] == 1)
		{
			_itlh.push_back(_itl[i]);
			_itlh[i].id = i;
		}
	}
	cout<<endl;

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


int l2_fastalm_mo(I_TRACK_LINK _itl,RESULTS _p)
{
	int D = _itl.xy_data.rows;
	int N =_itl.xy_data.cols;
	Mat Omega  = Mat();
	int nr = ceil((double)N/(D+1))*D;
	int nc = N - ceil((double)N/(D+1))+1;

	int defMaxIter = 1000;
	int defTol = 0.0000001;

	if(_itl.omega.empty())
		Omega = Mat::ones(1,N,CV_8U);
	else
		_itl.omega.copyTo(Omega);

	if(_p.nr ==INF)
		 _p.nr = nr;
	else
		nr =  _p.nr;

	if(_p.nc ==INF)
		_p.nc = nc;
	else
		nc =  _p.nc;

	if(_p.nc ==INF)
		_p.nc = nc;
	else
		nc =  _p.nc;

	//to be continue
	//next assignment

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

		l2_fastalm_mo(_itl,Lambda);	Mat matH=Mat();
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

	}


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
			compute_itl_similarity_matrix(itlh,params);
<<<<<<< .mine			dN--;
=======			Nnew = itlh.size();
			dN = N - Nnew;
			N = Nnew;
>>>>>>> .theirs		}

		
	}
	
	return 1;
}


