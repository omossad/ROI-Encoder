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
#include <x265.h>
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
static unsigned int frames ;

//***DIFFERS PER TECHNIQUE (CAVE_ROI uses one kernel while CAVE_WEIGHTED uses three kernels)
static cl::Kernel kernel[MAX_KERNELS];
static cl::Program program[MAX_KERNELS];
static cl::Buffer inp_buf[MAX_BUFS];
static cl::Buffer out_buf[MAX_BUFS];	
static cl_int err;	
static int reversed;

//****USED BY CAVE LAMBDA

#define SW 120
#define GOP 90


//****USED BY CAVE ROI

#define MIN_QP 0
#define MAX_QP 51
unsigned long long int* temp;
static bool canRead = 0;

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
static FILE* raw_yuv;
static FILE* encoded;
static float last_size=0.0f;
static unsigned long long int SCALE= 10000000000000;
static x265_encoder* vencoder;


string raw_path;
string roi_path;
string encoded_path;
string folderIn;
string folderOut;
string ga_logfile;
string slash;



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


static int vencoder_init() {

		x265_param params;
		x265_param_default(&params);				
		x265_param_default_preset(&params, "ultrafast", "zerolatency");
		char tmpbuf[500];			
		//if(mode!=BASE_ENCODER_){
			int ret=x265_param_parse(&params, "qp", "22");
			params.rc.ipFactor = pow(2.0,1.0/12.0);//to make sure that I frames have same QP as P frames			
			ret = x265_param_parse(&params, "aq-mode", "1");						
			//ret = x265_param_parse(&params, "no-wpp", "1");
			//ret = x265_param_parse(&params, "frame-threads", "4");
			//ret = x265_param_parse(&params, "slices", "4");
			//ret = x265_param_parse(&params, "lookahead-slices", "0");
			

			/*int ret = x265_param_parse(&params, "bitrate", "256");
			ret = x265_param_parse(&params, "vbv-maxrate", "256");
			ret = x265_param_parse(&params, "vbv-minrate", "256");
			ret = x265_param_parse(&params, "vbv-bufsize", "8.53");
			ret = x265_param_parse(&params, "strict-cbr", "");*/
	
		//}
		/*else
		{	
			string tmp = std::to_string((_Longlong) bitrate/1024); 		
			x265_param_parse(&params, "bitrate", tmp.c_str());					
			x265_param_parse(&params, "vbv-maxrate", tmp.c_str());			
				tmp = std::to_string((_Longlong) bitrate/(2*1024));
			x265_param_parse(&params, "vbv-bufsize", tmp.c_str());	
			x265_param_parse(&params, "strict-cbr", "1");							
		}*/
		string tmp = std::to_string((_Longlong) fps); 
		string intra = std::to_string((_Longlong)GOP);
		ret = x265_param_parse(&params, "keyint", intra.c_str());
		ret = x265_param_parse(&params, "intra-refresh", "1");
		ret = x265_param_parse(&params, "fps", tmp.c_str());
		ret = x265_param_parse(&params, "ref", "1");
		ret = x265_param_parse(&params, "me", "dia");
		ret = x265_param_parse(&params, "merange", "16");
		ret = x265_param_parse(&params, "bframes", "0");		
		params.logLevel = X265_LOG_INFO;
		params.internalCsp = X265_CSP_I420;
		params.sourceWidth= realWidth;
		params.sourceHeight = realHeight;		
		params.bRepeatHeaders = 1;
		params.bAnnexB = 1;
		ret = x265_param_parse(&params, "sar", "1");
		vencoder = x265_encoder_open(&params);	
	//}
	return 0;
}



static bool vencoder_encode(void * frame) {	
	
		x265_encoder *encoder = NULL;
		int pktbufsize = 0;
		int64_t x265_pts = 0;
		x265_param params;
		x265_encoder_parameters(vencoder, &params);

		x265_picture pic_in, pic_out = { 0 };
		x265_nal *nal;
		unsigned int i, size;
		uint32_t nnal;

		if (frame != NULL) {
			x265_picture_init(&params, &pic_in);
			x265_picture_init(&params, &pic_out);
			pic_out.colorSpace = X265_CSP_I420;

			pic_in.colorSpace = X265_CSP_I420;
			pic_in.stride[0] = realWidth;
			pic_in.stride[1] = realWidth >> 1;
			pic_in.stride[2] = realWidth >> 1;
			pic_in.planes[0] = frame;
			pic_in.planes[1] = (uint8_t *)(pic_in.planes[0]) + realWidth * realHeight;
			pic_in.planes[2] = (uint8_t *)(pic_in.planes[1]) + ((realWidth*realHeight) >> 2);
			pic_in.quantOffsets = QP_out;
		}

		clock_t begin = clock();
		size = x265_encoder_encode(vencoder, &nal, &nnal, &pic_in, &pic_out);
		clock_t end = clock();
		double temp = diffclock(end, begin);
		//if (frame == NULL && size == 0)
		//	return true;//flush ended
		if (size > 0) {
			for (i = 0; i < nnal; i++) {
				fwrite(nal[i].payload, sizeof(uint8_t), nal[i].sizeBytes, encoded);
			}
		}
	
		return false;

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
	realWidth =  atoi(config["width"].c_str());	
	realHeight = atoi(config["height"].c_str());
	raw_path = config["raw_path"]+slash+"raw_"+config["width"]+"_"+config["height"]+".yuv";
	ifstream file( raw_path, ios::binary | ios::ate);
	maxFrames = file.tellg()/(realWidth * realHeight * 1.5f);	
	file.close();
	heightDelta = (((realHeight / 2) + X265_LOWRES_CU_SIZE - 1) >> X265_LOWRES_CU_BITS);//will get the number of 16X16 blocks in the height direction for x265
	widthDelta = (((realWidth / 2) + X265_LOWRES_CU_SIZE - 1) >> X265_LOWRES_CU_BITS);

	pixelsPerBlock = (float)(CU_SIZE * CU_SIZE);
	QP_out = (float *) calloc(pow(upSampleRatio,2)*widthDelta * heightDelta,sizeof(float));
	temp = (unsigned long long int *)calloc(widthDelta*heightDelta,sizeof(unsigned long long int));
					

	
	size_t found=raw_path.find_last_of("/\\")-1;
	folderIn = raw_path.substr(0,found);
	folderOut = raw_path.substr(0,found);
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
	encoded=fopen(encoded_path.c_str(),"wb");		

	return 0;		
}




static void readQPs() {
	string file_idx = std::to_string((_Longlong)(frames / ROI_UPDATE_STEP));
	string qp_path = folderIn + slash + "qps" + slash + "qp" + file_idx + ".txt";
	//cout << qp_path.c_str() << endl;
	std::ifstream qp_infile(qp_path.c_str());
	int qp_val;
	unsigned int block_ind = 0;
	while (qp_infile >> qp_val) {
		QP_out[block_ind] = qp_val;
		block_ind++;
	}
	qp_infile.close();
}



static void vrc_start() {	
	void * frame = calloc(1.5f*realHeight*realWidth,sizeof(char));
			while(frames<maxFrames) {										
				fseek(raw_yuv,sizeof(char)*frames*1.5f*realHeight*realWidth,0);
				fread(frame,sizeof(char),1.5f*realHeight*realWidth,raw_yuv);	
				readQPs();
				vencoder_encode(frame);
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
	vencoder_init();
	vrc_start();
	//ga_error("bitrate : %.2f", (totalBits / (maxFrames/fps))/1024);
	return 0;
}

