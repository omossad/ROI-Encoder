#include <stdio.h>
#include <stdarg.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include <math.h>
#include <time.h>
#include "CL/cl.hpp"
using namespace std;

typedef long long _Longlong;
#pragma comment(linker, "/STACK:2000000")
#pragma comment(linker, "/HEAP:2000000")
//static bool KVZ = 0;
#ifdef	_WIN32
#include<windows.h>
#else
#define	BACKSLASHDIR(fwd, back)	fwd
#include <sys/stat.h>
#endif

#define	ROI_UPDATE_STEP 1
#define QP_BASE 22
#define X265_LOWRES_CU_SIZE   8
#define X265_LOWRES_CU_BITS   3
#define CU_SIZE 16 // for x265 as well as x264
#define MAX_KERNELS 3
#define MAX_BUFS 10
#define ISCALE 1
#define DISTANCE 75 //average viewing distance on PC

static double K = 3;
static unsigned int upSampleRatio = CU_SIZE / 16;


enum OBJECT_IMPORTANCE{
	HIGH,
	MEDIUM,
	LOW
};

typedef struct ROITuple_s {
	int x;//top left x
	int y;//top left y
	int width;
	int height;
	int category;
}	ROITuple_t;//For debug purposes

double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(diffticks*1000)/CLOCKS_PER_SEC;
	return diffms;
} 


//***SHARED AMONG ALL TECHNIQUES**
static std::vector<cl::Platform> platforms;
static std::vector<cl::Device> devices;	
static cl::CommandQueue queue;
static cl::Context context;
static float diagonal;
static float * weights;
static unsigned int frames ;
static float importance[3] = {1.0f,0.95f,0.5f};//based on OBJECT_IMPORTANCE enum

//***DIFFERS PER TECHNIQUE (CAVE_ROI uses one kernel while CAVE_WEIGHTED uses three kernels)
static cl::Kernel kernel[MAX_KERNELS];
static cl::Program program[MAX_KERNELS];
static cl::Buffer inp_buf[MAX_BUFS];
static cl::Buffer out_buf[MAX_BUFS];	
static cl_int err;	
static int reversed;

//****USED BY CAVE LAMBDA
static double * bitsPerBlock;
static double ALPHA=3.2003f;
static double BETA=-1.367f;
static double C1=4.2005f;
static double C2=13.7112f;
static double m_alphaUpdate = 0.1f;
static double m_betaUpdate  =0.05f;
static double MIN_LAMBDA = 0.1f;
static double MAX_LAMBDA = 10000.0f;
static double FRAME_LAMBDA ;
static double totalBitsUsed;
static double bitsTotalGroup ;
static double avg_bits_per_pic ;
static double old_avg_bits_per_pic ;
static double avgLambda = 0.0f;
static double sumRate = 0.0f;
static double totalBits=0.0f;
#define SW 120
#define GOP 90
static double lambda_buf_occup = 0.0;
static double * base_MAD;
static double sum_MAD;


//****USED BY CAVE ROI
#define ADJUSTMENT_FACTOR       0.60f
#define HIGH_QSTEP_ALPHA        4.9371f
#define HIGH_QSTEP_BETA         0.0922f
#define LOW_QSTEP_ALPHA         16.7429f
#define LOW_QSTEP_BETA          -1.1494f
#define HIGH_QSTEP_THRESHOLD    9.5238f
#define MIN_QP 0
#define MAX_QP 51
unsigned long long int* temp;
static float BIAS_ROI = 3.0f;
static float  upper;
static float  lower;
static float m_paramLowX1=LOW_QSTEP_ALPHA;
static float m_paramLowX2=LOW_QSTEP_BETA;
static float m_paramHighX1 =HIGH_QSTEP_ALPHA;
static float m_paramHighX2= HIGH_QSTEP_BETA;
static float m_Qp;
static float m_remainingBitsInGOP;
static int m_initialOVB;
static float m_occupancyVB,m_initialTBL,m_targetBufLevel;
static bool canRead = 0;
static float filter_mean[3][3] = { {1.0f / 12.0f,1.0f / 12.0f,1.0f / 12.0f}, {1.0f / 12.0f,1.0f / 3.0f,1.0f / 12.0f}, {1.0f / 12.0f,1.0f / 12.0f,1.0f / 12.0f} };

//Control variables
static unordered_map<string,string> config;
static bool reassign = 0;
static unsigned int realWidth = 0;
static unsigned int realHeight = 0;
static unsigned int widthDelta = 0;
static unsigned int heightDelta = 0;
static float pixelsPerBlock = 0;
static float bitrate = 0.0f;
static unsigned int fps = 0;
//static unsigned int period = 0;
static float * QP;
static float * QP_out;
static float * roi;
static float*   ROI;
static unsigned long long int maxFrames;
static std::vector<ROITuple_t> ROIs;
static FILE* raw_yuv;
static FILE* encoded;
static float last_size=0.0f;
static unsigned long long int SCALE= 10000000000000;


string raw_path;
string roi_path;
string encoded_path;
string folderIn;
string folderOut;
string ga_logfile;
string slash;


static int roundRC(float d)
{
    return static_cast<int>(d + 0.5);
}

/*!
 *************************************************************************************
 * \brief
 *    map QP to Qstep
 *
 *************************************************************************************
*/


int ga_error(const char *fmt, ...) {
	char msg[4096];
	va_list ap;	
	va_start(ap, fmt);
#ifdef ANDROID
	__android_log_vprint(ANDROID_LOG_INFO, "ga_log.native", fmt, ap);
#endif
	vsnprintf(msg, sizeof(msg), fmt, ap);
	va_end(ap);
#ifdef __APPLE__
	syslog(LOG_NOTICE, "%s", msg);
#endif
	FILE *fp;	
	fp=fopen(ga_logfile.c_str(), "at");
	fprintf(fp, "%s", msg);
	fclose(fp);	
	return -1;
}

static void initCL(){
	//ga_error("Initialize CL\n");
	clock_t begin=clock();
	err=cl::Platform::get(&platforms);
	if(err!=CL_SUCCESS)
	{
		ga_error("Platform err:%d\n",err);
		return;
	}
	string platform_name;
	string device_type;		
	//ga_error("Number of Platforms Available:%d\n",platforms.size());	
	platforms[0].getInfo(CL_PLATFORM_NAME,&platform_name);	
	//ga_error("Platform Used:%s\n",platform_name.c_str());
	err=platforms[0].getDevices(CL_DEVICE_TYPE_ALL,&devices);
	if(err!=CL_SUCCESS)
	{
		ga_error("Device err:%d\n",err);
		return;
	}	
	ga_error("Number of Devices Available:%d\n",devices.size());
	err=devices[0].getInfo(CL_DEVICE_NAME,&device_type);
	if(err!=CL_SUCCESS)
		ga_error("Type of device\n");
	else{		
		//ga_error("Type of Device Used: %s\n",device_type.c_str());
	}
	context=cl::Context(devices,NULL,NULL,NULL,&err);
	if(err!=CL_SUCCESS)
		ga_error("Context err:%d\n",err);
	queue=cl::CommandQueue(context,devices[0],NULL,&err);
	if(err!=CL_SUCCESS)
		ga_error("Command Queue err:%d\n",err);
	clock_t end=clock();	
	//ga_error("Time Constructor: %f\n",diffclock(end,begin));	
}

std::string LoadKernel (const char* name)
{
	char srcPath[1024];	
	sprintf(srcPath,name);	
	std::ifstream in (srcPath);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	//cout<<result<<endl;
	int index = result.find("?w?");    
	string buf;	
	buf=std::to_string((_Longlong)realWidth);
    result=result.replace(index, 3,buf);
	index = result.find("?h?");    		
	buf=std::to_string((_Longlong)realHeight);
    result=result.replace(index, 3,buf);
	index = result.find("?c?");    		
	buf=std::to_string((_Longlong)(int)(log((double)CU_SIZE)/log(2.0)));
    result=result.replace(index, 3,buf);
	index = result.find("?bw?");    		
	buf=std::to_string((_Longlong)widthDelta);
    result=result.replace(index, 4,buf);
	index = result.find("?bh?");    	
	buf=std::to_string((_Longlong)heightDelta);	
    result=result.replace(index, 4,buf);
	index = result.find("?s?");    		
	buf=std::to_string((_Longlong)SCALE);
    result=result.replace(index, 3,buf);
	//ga_error(result.c_str());
	return result;
}

static void loadCL(string name,int idx,string signature){
	//ga_error("Load Program\n");
	clock_t begin=clock();
	std::string src = LoadKernel(name.c_str());
	cl::Program::Sources source(1,make_pair(src.c_str(),src.size()));
	program[idx]=cl::Program(context,source,&err);
	if(err!=CL_SUCCESS)
		ga_error("Program err:%d\n",err);
	err=program[idx].build(devices);
	if(err!=CL_SUCCESS)
		ga_error("Build Error err:%d\n",err);
	//ga_error("done building program\n");
	clock_t end=clock();		
	//ga_error("Time Build Program: %f\n",diffclock(end,begin));	
	//ga_error("Build Status: %d\n" , program[idx].getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[0]));		
	//ga_error("Build Options: %d\n", program[idx].getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[0]));		
	//if(err!=CL_SUCCESS)		
	//	ga_error("Build Log: %s\n" , program[idx].getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]));	
	//ga_error("Create Kernel\n");
	
	
	begin=clock();
	kernel[idx]=cl::Kernel(program[idx],signature.data(),&err);
	if(err!=CL_SUCCESS)
		ga_error("Create Kernel Error err:%d\n",err);
	end=clock();
	//ga_error("Time Create Kernel %f\n",diffclock(end,begin));
}






static int vrc_init() {	
	#ifdef	_WIN32	
	slash = "\\\\";
	#else
	slash = "/";
	#endif

	fps =  atoi(config["fps"].c_str());	
	bitrate = atoi(config["bitrate"].c_str());
	K = strtod(config["decay"].c_str(), NULL);
	m_remainingBitsInGOP = bitrate / fps * GOP;
	lambda_buf_occup = 0.5 * bitrate / fps;
	bitsTotalGroup = max(200.0f, (((1.0f * bitrate / fps) - 0.5 * lambda_buf_occup / fps) * GOP));
	//bitsTotalGroup = max(200.0f, (((1.0f * bitrate / fps)*(frames + SW) - totalBits)*GOP) / SW);
	avg_bits_per_pic = (float)(bitrate / (1.0f * (float)fps));
	old_avg_bits_per_pic = avg_bits_per_pic;		
	realWidth =  atoi(config["width"].c_str());	
	realHeight = atoi(config["height"].c_str());
	raw_path = config["raw_path"]+slash+"raw_"+config["width"]+"_"+config["height"]+".yuv";
	ifstream file( raw_path, ios::binary | ios::ate);
	maxFrames = file.tellg()/(realWidth * realHeight * 1.5f);	
	file.close();
	diagonal = (float)sqrt(pow(realWidth * 1.0, 2) + pow(realHeight * 1.0, 2));

	heightDelta = (((realHeight / 2) + X265_LOWRES_CU_SIZE - 1) >> X265_LOWRES_CU_BITS);//will get the number of 16X16 blocks in the height direction for x265
	widthDelta = (((realWidth / 2) + X265_LOWRES_CU_SIZE - 1) >> X265_LOWRES_CU_BITS);

	pixelsPerBlock = (float)(CU_SIZE * CU_SIZE);
	QP = (float *) calloc(widthDelta * heightDelta,sizeof(float));
	QP_out = (float *) calloc(pow(upSampleRatio,2)*widthDelta * heightDelta,sizeof(float));
	bitsPerBlock = (double *)calloc(heightDelta * widthDelta, sizeof(double));
	weights = (float *)calloc(heightDelta * widthDelta, sizeof(float));
	ROI = (float *)calloc(heightDelta * widthDelta, sizeof(float)); //predict QPs using complexity of depth
	base_MAD = (double *)calloc(heightDelta * widthDelta, sizeof(double)); //predict QPs using complexity of depth
	temp = (unsigned long long int *)calloc(widthDelta*heightDelta,sizeof(unsigned long long int));
					
	/*rois=fopen( "C:\\Users\\mhegazy\\Desktop\\rois.txt","ab");
	qp=fopen( "C:\\Users\\mhegazy\\Desktop\\qp.txt","ab");
	importance=fopen( "C:\\Users\\mhegazy\\Desktop\\importance.txt","ab");
	bits=fopen( "C:\\Users\\mhegazy\\Desktop\\bits.txt","ab");
	bits_actual=fopen( "C:\\Users\\mhegazy\\Desktop\\bits_actual.txt","ab");*/

	
	size_t found=raw_path.find_last_of("/\\")-1;
	folderIn = raw_path.substr(0,found);
	folderOut = raw_path.substr(0,found);

		double m_seqTargetBpp = bitrate / (fps * realWidth * realHeight);

		m_alphaUpdate = 0.01;
		m_betaUpdate  = 0.005;

	folderOut = folderOut + slash + "output";


	 #if defined(_WIN32)
    CreateDirectory(folderOut.c_str(),NULL);
     #else 
    mkdir(folderOut.c_str(), 0777); 
     #endif		

	folderOut = folderOut +slash+config["bitrate"];
	#if defined(_WIN32)
    CreateDirectory(folderOut.c_str(),NULL);
     #else 
    mkdir(folderOut.c_str(), 0777); 
     #endif	
	encoded_path = folderOut +slash+ "enc.mp4";
	ga_logfile = folderOut +slash+ "log.txt";
	FILE *tmp = fopen(ga_logfile.c_str(), "wb");
	fclose(tmp);//just to remove old contents
	raw_yuv=fopen(raw_path.c_str(),"rb");	
	return 0;		
}


//This function should load the raw frame, the user input and try to predict the ROI (e.g. using a neural network model that was trained on different ROIs)
//for now assume the ROI is in the middle of the screen
static void loadROIs(){
	if(frames%ROI_UPDATE_STEP==0){
		ROIs.clear();
		memset(ROI,0,widthDelta *heightDelta * sizeof(float));
		string file_idx=std::to_string((_Longlong)(frames/ROI_UPDATE_STEP));		
		std::ifstream infile(folderIn+slash+"rois"+slash+"roi"+file_idx+".txt");
		int category;
		double x,y,w,h;		
		while(infile >> category >> x >> y >> w >> h){
			ROITuple_s r;
			r.x= (x-w/2)*realWidth;
			r.y= (y-h/2)*realHeight;
			r.width= (w)*realWidth;
			r.height= (h)*realHeight;
			r.category = category;
			ROIs.push_back(r);				
		}
		infile.close();		
	}	
}

FILE *tmp_qp;
static void writeQPs() {
	string file_idx = std::to_string((_Longlong)(frames / ROI_UPDATE_STEP));
	string qp_path = folderIn + slash + "qps" + slash + "qp" + file_idx + ".txt";
	//cout << qp_path.c_str() << endl;
	std::ofstream tmp_qp(qp_path.c_str());
	unsigned int block_ind = 0;
	for (unsigned int x = 0; x < heightDelta; x++) {
		for (unsigned int y = 0; y < widthDelta; y++) {
			tmp_qp << QP_out[block_ind] << endl;
			block_ind++;
		}
	}
	tmp_qp.close();

	//tmp_qp = fopen(qp_path.c_str(), "wb");
	//fprintf(tmp_qp, "%s", msg.c_str());
	//fclose(tmp_qp);
}





static float CLIP(float min_, float max_, float value)
{
	return max((min_), min((max_), value));
}

static void clipLambda(double * lambda)
{
#ifdef _WIN32
	if (_isnan(*lambda))
#else
	if (isnan(*lambda))
#endif
	{

		*lambda = MAX_LAMBDA;
	}
	else
	{
		*lambda = CLIP(MIN_LAMBDA, MAX_LAMBDA, (*lambda));
	}
}

static float QPToBits(int QP)
{
	float lambda = exp((QP - C2) / C1);
	float bpp = (float)pow((lambda / ALPHA)*1.0f, 1.0f / BETA);
	return bpp * (float)pow(64.0f, 2.0f);
}

static int LAMBDA_TO_QP(float lambda)
{
    return (int)CLIP(0.0f, 51.0f, (float)roundRC(C1 * log(lambda) + C2));
}

//This function should load the raw frame, the user input and try to predict the ROI
static void updateROIs(){		
	loadROIs();//assume for now ROIs are in a file
	memset(ROI,0,widthDelta *heightDelta * sizeof(float));
	//simulate blue masks in the paper 45% is high importance and 30% is medium importance
		for(unsigned int r=0;r<ROIs.size();r++){
			unsigned int xTop=ROIs[r].x;
			unsigned int yTop=ROIs[r].y;
			unsigned int xBottom=xTop + ROIs[r].width;
			unsigned int yBottom=yTop + ROIs[r].height;
			xTop = xTop / CU_SIZE;
			yTop = yTop / CU_SIZE;
			xBottom = xBottom / CU_SIZE;
			yBottom = yBottom / CU_SIZE; 		
			for (unsigned int j = yTop; j <= yBottom; j++)
			{
				for (unsigned int k = xTop; k <= xBottom; k++)
				{				
					if(ROIs[r].category==HIGH)
						ROI[k+j*widthDelta]=0.45;				
					else if(ROIs[r].category==MEDIUM)
						ROI[k+j*widthDelta]=0.3;
					else if (ROIs[r].category == LOW)
						ROI[k + j * widthDelta] = 0.15;
				}			
			}
		}
	
	

	//fwrite(ROI,sizeof(bool),widthDelta*heightDelta,rois);
	//ga_error("finished roi assignment\n");
}







//****************************************************************LAMBDA DOMAIN FUNCTIONS BEGIN******************************************************
//run on CPU as in our calculations we don't need to process every pixel	
// std::sort with inlining
int compare(const void* elem1, const void* elem2)
{
	if (*(const float*)elem1 < *(const float*)elem2)
		return -1;
	return *(const float*)elem1 > *(const float*)elem2;
}

static void meanFilter() {
	int block_ind = 0;
	float sumWeights = 0;
	for (int i = 0;i<heightDelta;i++) {
		for (int j = 0;j<widthDelta;j++) {
			float val = 0.0;
			for (unsigned int a = 0; a < 3; a++)
			{
				for (unsigned int b = 0; b < 3; b++)
				{
					int jn = min(widthDelta - 1, max(0, j + b - 1));
					int in = min(heightDelta - 1, max(0, i + a - 1));
					int index = jn + in * widthDelta;
					val = val + 1.0f* weights[index] * filter_mean[a][b];
				}
			}
			weights[block_ind] = val;
			sumWeights = sumWeights + val;
			block_ind++;
		}
	}
}


static void updateDistanceCAVELambda() {
	float sumDistance = 0.0f;
	unsigned int block_ind = 0;

	for (unsigned int x = 0;x < heightDelta;x++) {
		for (unsigned int y = 0;y < widthDelta;y++) {
			
			if (ROI[block_ind]>0.0) {
				float e = 1/(K*ROI[block_ind]);
				weights[block_ind] = exp(-1.0*e);
			}
			else {
				float sumDist = 0.0f;
				int xpixIndex = x * CU_SIZE + CU_SIZE / 2;
				int ypixIndex = y * CU_SIZE + CU_SIZE / 2;
				for (unsigned int r = 0;r<ROIs.size();r++) {
					int xMid = ROIs[r].x + ROIs[r].width / 2;
					int yMid = ROIs[r].y + ROIs[r].height / 2;
					float dist = max(1.0f, (float)sqrt(pow(xpixIndex - yMid, 2.0) + pow(ypixIndex - xMid, 2.0)));					
					float e = (K*dist/ diagonal) / (importance[ROIs[r].category]) ;
					sumDist = sumDist + exp(-1.0*e);
				}

				if (ROIs.size() > 0)//ROI and CAVE
					weights[block_ind] = sumDist / ROIs.size();
			}
			sumDistance = sumDistance + weights[block_ind];
			block_ind++;
		}	
	}

	block_ind = 0;
	for(unsigned int x=0;x<heightDelta;x++){
		for(unsigned int y=0;y<widthDelta;y++){	
	
			weights[block_ind]=weights[block_ind]/sumDistance;		
			block_ind++;		
		}
	}
	meanFilter();

}











static void calcQPLambda()
{
	if (frames == 0)
	{
		double targetbppFrame = (1.0f * avg_bits_per_pic * (ISCALE)) / (1.0f * realWidth * realHeight);
		FRAME_LAMBDA = ALPHA * pow(targetbppFrame, BETA);
		clipLambda(&FRAME_LAMBDA);
	}
	//else
	//	updateParameters();//returns the actual number of bits used for each block in the previous frame


	double bitsAlloc = avg_bits_per_pic;
	if (frames % GOP == 0)
		bitsAlloc = bitsAlloc * ISCALE;
	
	double qp_frame = LAMBDA_TO_QP(ALPHA*pow(bitsAlloc / (realWidth*realHeight),BETA));
	double qp_delta = 0.0;
	double extra = 0.0;
	double sumSaliencyImportantAreas = 0.0;
	double sumSaliency = 0.0;
	int reassigned = 0;
	avgLambda = 0.0f;
    unsigned int block_ind=0;	
	double avg_qp = 0;			
	double sum_non_roi = 0;
	for (unsigned int i = block_ind; i < widthDelta*heightDelta; i++)
	{		
			sumSaliency = sumSaliency + weights[block_ind];			
			double assigned = max(1.0f, 1.0f / (1.0f*widthDelta*heightDelta) * bitsAlloc);
			double assignedW = weights[block_ind] * bitsAlloc;
			double targetbpp=0.0f;
            
			targetbpp = (assignedW) / (pixelsPerBlock);
			if (ROI[block_ind] == 0) {
				sum_non_roi++;
			}
			double lambdaConst = ALPHA * pow(targetbpp, BETA);//Kiana's idea of using the updated ALPHA/BETA for the whole frame 					
			avgLambda = avgLambda+log(lambdaConst); 
			double temp = (double)LAMBDA_TO_QP(lambdaConst);
			
			qp_delta = qp_delta + temp - qp_frame;	
			QP[block_ind] = temp -QP_BASE;

			avg_qp = avg_qp + QP[block_ind]+QP_BASE;
			bitsPerBlock[block_ind] = min(QPToBits((int)QP[block_ind]),(double)assigned);
			block_ind++;		
	}
	if (qp_delta != 0) {		
			while (qp_delta < 0) {//overshoot
				for (block_ind = 0;block_ind < widthDelta*heightDelta && qp_delta<0;block_ind++) {
					if (ROI[block_ind] == 0)//increase non-ROI QPs -> keep ROI gains intact
						QP[block_ind] = min(51-QP_BASE, QP[block_ind] + 1);//make sure QP doesn't exceed 51
					qp_delta = qp_delta + 1;
				}
				//if qp_delta is still below 0 we won't increase the QPs inside the ROIs in order not to sacrifice their quality
			}
		
	}
	ga_error("qp frame: %.2f , qp delta : %.2f avg qp:%.2f\n" ,qp_frame, qp_delta, avg_qp / (widthDelta*heightDelta));
	avgLambda = exp(avgLambda / (widthDelta*heightDelta));	
	old_avg_bits_per_pic = avg_bits_per_pic;    
}
//****************************************************************LAMBDA DOMAIN FUNCTIONS END******************************************************

static void upSample(float * qps, float * dst){
	if (upSampleRatio == 1) {
		memcpy(dst, qps, sizeof(float)*widthDelta*heightDelta);
		return;
	}
	unsigned int block_ind =0;
	for(unsigned int i=0,iSmall=0;i<heightDelta;i++,iSmall=iSmall+ upSampleRatio){
		for(unsigned int j=0,jSmall=0;j<widthDelta;j++,jSmall=jSmall+ upSampleRatio){
			float val=qps[block_ind];
			dst[jSmall+iSmall*widthDelta*upSampleRatio]=dst[jSmall+iSmall*widthDelta*upSampleRatio +1]=dst[jSmall+(iSmall+1)*widthDelta*upSampleRatio]=dst[jSmall+(iSmall+1)*widthDelta*upSampleRatio +1]=val;
			block_ind++;
		}
	}
}


static void vrc_start() {	
	void * frame = calloc(1.5f*realHeight*realWidth,sizeof(char));
			while(frames<maxFrames) {										
				//fseek(raw_yuv,sizeof(char)*frames*1.5f*realHeight*realWidth,0);
				//fread(frame,sizeof(char),1.5f*realHeight*realWidth,raw_yuv);
				updateROIs();
				updateDistanceCAVELambda();		
				calcQPLambda();
				upSample(QP,QP_out);
				writeQPs();
				frames++;				
			}				
	free(frame);

}





int main(int argc, char *argv[]) {
	char * configFileName = (char *)argv[1];//config file contains a config line with the following format: <key1>=<val1>:<key2>=<val2> , expected keys are fps, raw_video_path, width, height, length
	//config["bitrate"]=(char *)argv[2];
	//K = strtod((char *)argv[5],NULL);	
	string f(configFileName);
	config["raw_path"] = f;
	f = f + "conf.txt";
	ifstream infile(f);
	if (infile.good())
	{		
		std::string param;
		while (std::getline(infile, param)) {
			std::stringstream paramStream(param);
			std::string key;
			std::string val;
			std::getline(paramStream, key, '=');
			std::getline(paramStream, val, '=');
			config[key]=val;		
			cout<<key<<"="<<val<<endl;
		}	
		infile.close();
	}
	vrc_init();
	vrc_start();
	ga_error("bitrate : %.2f", (totalBits / (maxFrames/fps))/1024);
	return 0;
}

