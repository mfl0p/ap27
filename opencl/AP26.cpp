/* 
	AP26 GPU OpenCL application
 	Bryan Little
	with contributions by Yves Gallot
 	April 15, 2023				*/


#include <cinttypes>

#include "const.h"

#include "boinc_api.h"
#include "boinc_opencl.h"

#include "simpleCL.h"

// ocl kernels
#include "clearok.h"
#include "clearokok.h"
#include "clearn.h"
#include "checkn.h"
#include "offset.h"
#include "setupn.h"
#include "setupokok.h"
#include "setupok.h"
#include "sieve.h"
#include "sieve_nv.h"

#define numn59s 137375320
#define halfn59s 68687660
#define numn43s	10840
#define numOK 23693
#define sol 10240

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

#define STATE_FILENAME_A "AP26-state.a.txt"
#define STATE_FILENAME_B "AP26-state.b.txt"
#define RESULTS_FILENAME "SOL-AP26.txt"

#define MINIMUM_AP_LENGTH_TO_REPORT 20
#define MAXINTV 2000000000

/* Global variables */
int KMIN, KMAX, K_DONE, K_COUNT;
uint32_t totalaps;
bool write_state_a_next;
uint32_t numn;
uint32_t cksum;
uint64_t last_trickle;
time_t last_ckpt;

sclHard hardware;

sclSoft offset;
sclSoft checkn;
sclSoft setupokok;
sclSoft setupok;
sclSoft sieve;
sclSoft setupn;
sclSoft clearok;
sclSoft clearokok;
sclSoft clearn;

uint64_t *n43_h;
uint64_t *sol_val_h;
int *sol_k_h;
int *counter_h;
cl_mem n_result_d = NULL;
cl_mem counter_d = NULL;
cl_mem OKOK_d = NULL;
cl_mem OK_d = NULL;
cl_mem offset_d = NULL;
cl_mem sol_k_d = NULL;
cl_mem sol_val_d = NULL;
cl_mem n43_d = NULL;
cl_mem n59_0_d = NULL;
cl_mem n59_1_d = NULL;

FILE *results_file = NULL;


void handle_trickle_up(){

	if(boinc_is_standalone()) return;

	uint64_t now = (uint64_t)time(NULL);

	if( (now-last_trickle) > 86400 ){	// Once per day

		last_trickle = now;

		double progress = boinc_get_fraction_done();
		double cpu;
		boinc_wu_cpu_time(cpu);
		APP_INIT_DATA init_data;
		boinc_get_init_data(init_data);
		double run = boinc_elapsed_time() + init_data.starting_elapsed_time;

		char msg[512];
		sprintf(msg, "<trickle_up>\n"
			    "   <progress>%lf</progress>\n"
			    "   <cputime>%lf</cputime>\n"
			    "   <runtime>%lf</runtime>\n"
			    "</trickle_up>\n",
			     progress, cpu, run  );
		char variety[64];
		sprintf(variety, "ap26_progress");
		boinc_send_trickle_up(variety, msg);
	}

}


/*
	tests primality of each term of the AP sequence
	test is good to 2^64-1
*/


uint64_t invert(uint64_t p)
{
	uint64_t p_inv = 1, prev = 0;
	while (p_inv != prev) { prev = p_inv; p_inv *= 2 - p * p_inv; }
	return p_inv;
}


uint64_t montMul(uint64_t a, uint64_t b, uint64_t p, uint64_t q)
{
	unsigned __int128 res;

	res  = (unsigned __int128)a * b;
	uint64_t ab0 = (uint64_t)res;
	uint64_t ab1 = res >> 64;

	uint64_t m = ab0 * q;

	res = (unsigned __int128)m * p;
	uint64_t mp = res >> 64;

	uint64_t r = ab1 - mp;

	return ( ab1 < mp ) ? r + p : r;
}


uint64_t add(uint64_t a, uint64_t b, uint64_t p)
{
	uint64_t r;

	uint64_t c = (a >= p - b) ? p : 0;

	r = a + b - c;

	return r;
}


// initialize montgomery constants
void mont_init(uint64_t N, int & t, uint64_t & curBit, uint64_t & exp, uint64_t & nmo, uint64_t & q, uint64_t & one, uint64_t & r2){


	nmo = N-1;
	t = __builtin_ctzll(nmo);
	exp = N >> t;
	curBit = 0x8000000000000000;
	curBit >>= ( __builtin_clzll(exp) + 1 );
	q = invert(N);
	one = (-N) % N;
	nmo = N - one;
	uint64_t two = add(one, one, N);
	r2 = add(two, two, N);
	for (int i = 0; i < 5; ++i)
		r2 = montMul(r2, r2, N, q);	// 4^{2^5} = 2^64

}


bool strong_prp(int base, uint64_t N, int t, uint64_t curBit, uint64_t exp, uint64_t nmo, uint64_t q, uint64_t one, uint64_t r2)
{

	/* If N is prime and N = d*2^t+1, where d is odd, then either
		1.  a^d = 1 (mod N), or
		2.  a^(d*2^s) = -1 (mod N) for some s in 0 <= s < t    */


	uint64_t a = base;
	uint64_t mbase = montMul(a,r2,N,q);  // convert base to montgomery form

	a = mbase;

  	/* r <-- a^d mod N, assuming d odd */
	while( curBit )
	{
		a = montMul(a,a,N,q);

		if(exp & curBit){
			a = montMul(a,mbase,N,q);
		}

		curBit >>= 1;
	}

	/* Clause 1. and s = 0 case for clause 2. */
	if (a == one || a == nmo){
		return true;
	}

	/* 0 < s < t cases for clause 2. */
	for (int s = 1; s < t; ++s){

		a = montMul(a,a,N,q);

		if(a == nmo){
	    		return true;
		}
	}


	return false;
}




cl_mem sclMalloc( sclHard hardware, cl_int mode, size_t size ){
        cl_mem buffer;

        cl_int err;

        buffer = clCreateBuffer( hardware.context, mode, size, NULL, &err );
        if ( err != CL_SUCCESS ) {
                printf( "\nclMalloc Error\n" );

                fprintf(stderr,"OpenCL memory allocation error, restarting in 1 minute.\n");
                boinc_temporary_exit(60);
        }

        return buffer;
}


FILE *my_fopen(const char *filename, const char *mode)
{
	char resolved_name[512];

	boinc_resolve_filename(filename,resolved_name,sizeof(resolved_name));
	return boinc_fopen(resolved_name,mode);
}


void Progress(double prog){

	boinc_fraction_done(prog);

	if(boinc_is_standalone()){
		printf("Tests done: %.4f%%\n",prog*100.0);
	}

}


// BOINC checksum calculation, write to solution file, and close.
void write_cksum()
{
	uint64_t minmax = KMIN + KMAX;

	while(minmax > MAXINTV){
		minmax -= MAXINTV;
	}

	uint64_t bchecksum = (uint64_t)( (minmax << 32) | cksum);

	FILE* res_file = my_fopen(RESULTS_FILENAME,"a");

	if (res_file == NULL){
		fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
		exit(EXIT_FAILURE);
	}

	if (fprintf(res_file,"%016" PRIX64 "\n",bchecksum)<0){
		fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
		exit(EXIT_FAILURE);
	}

	fclose(res_file);

}


void write_state(int KMIN, int KMAX, int SHIFT, int K)
{
	FILE *out;

	if (write_state_a_next)
	{
		if ((out = my_fopen(STATE_FILENAME_A,"w")) == NULL)
			fprintf(stderr,"Cannot open %s !!!\n",STATE_FILENAME_A);
	}
	else
	{
		if ((out = my_fopen(STATE_FILENAME_B,"w")) == NULL)
			fprintf(stderr,"Cannot open %s !!!\n",STATE_FILENAME_B);
	}

	if (fprintf(out,"%d %d %d %d %u %u %" PRIu64 "\n",KMIN,KMAX,SHIFT,K,cksum,totalaps,last_trickle) < 0){
		if (write_state_a_next)
			fprintf(stderr,"Cannot write to %s !!! Continuing...\n",STATE_FILENAME_A);
		else
			fprintf(stderr,"Cannot write to %s !!! Continuing...\n",STATE_FILENAME_B);

		// Attempt to close, even though we failed to write
		fclose(out);
	}
	else
	{
		// If state file is closed OK, write to the other state file
		// next time round
		if (fclose(out) == 0) write_state_a_next = !write_state_a_next; 
	}
}

/* Return 1 only if a valid checkpoint can be read.
   Attempts to read from both state files,
   uses the most recent one available.
 */
int read_state(int KMIN, int KMAX, int SHIFT, int *K)
{
	FILE *in;
	bool good_state_a = true;
	bool good_state_b = true;
	int tmp1, tmp2, tmp3;
	int K_a, K_b;
	uint32_t cksum_a, cksum_b;
	uint32_t taps_a, taps_b;
	uint64_t trickle_a, trickle_b;

	// Attempt to read state file A
	if ((in = my_fopen(STATE_FILENAME_A,"r")) == NULL)
	{
		good_state_a = false;
	}
	else if (fscanf(in,"%d %d %d %d %u %u %" PRIu64 "\n",&tmp1,&tmp2,&tmp3,&K_a,&cksum_a,&taps_a,&trickle_a) != 7)
	{
		fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_A);
		good_state_a = false;
	}
	else
	{
		fclose(in);

		/* Check that KMIN KMAX SHIFT all match */
		if (tmp1 != KMIN || tmp2 != KMAX || tmp3 != SHIFT){
			good_state_a = false;
		}
	}

	// Attempt to read state file B
	if ((in = my_fopen(STATE_FILENAME_B,"r")) == NULL)
	{
		good_state_b = false;
	}
	else if (fscanf(in,"%d %d %d %d %u %u %" PRIu64 "\n",&tmp1,&tmp2,&tmp3,&K_b,&cksum_b,&taps_b,&trickle_b) != 7)
	{
		fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_B);
		good_state_b = false;
	}
	else
	{
		fclose(in);

		/* Check that KMIN KMAX SHIFT all match */
		if (tmp1 != KMIN || tmp2 != KMAX || tmp3 != SHIFT){
				good_state_b = false;
		}
	}

        // If both state files are OK, check which is the most recent
	if (good_state_a && good_state_b)
	{
		if (K_a > K_b)
			good_state_b = false;
		else
			good_state_a = false;
	}

        // Use data from the most recent state file
	if (good_state_a && !good_state_b)
	{
		*K = K_a;
		cksum = cksum_a;
		totalaps = taps_a;
		write_state_a_next = false;
		last_trickle = trickle_a;

		return 1;
	}
	if (good_state_b && !good_state_a)
	{
		*K = K_b;
		cksum = cksum_b;
		totalaps = taps_b;
		write_state_a_next = true;
		last_trickle = trickle_b;

		return 1;
	}

	// If we got here, neither state file was good
	return 0;
}


/* 
   Returns index j where:
   0<=j<k ==> f+j*d*23# is composite.
   j=k    ==> for all 0<=j<k, f+j*d*23# is a strong probable prime to base 2 only.
*/
int val_base2_ap26(int k, int d, uint64_t f)
{
	uint64_t N;
	int j;

	if (f%2==0)
		return 0;

	for (j = 0, N = f; j < k; j++){

		int t;
		uint64_t curBit, exp, nmo, q, one, r2;

		mont_init(N, t, curBit, exp, nmo, q, one, r2);

		if (!strong_prp(2, N, t, curBit, exp, nmo, q, one, r2))
			return j;

		N += (uint64_t)d*2*3*5*7*11*13*17*19*23;
	}

	return j;
}

/*
   Returns index j where:
   0<=j<k ==> f+j*d*23# is composite.
   j=k    ==> for all 0<=j<k, f+j*d*23# is prime

   test is good to 2^64-1
*/
int validate_ap26(int k, int d, uint64_t f)
{
	uint64_t N;
	int j;

	const int base[12] = {2,3,5,7,11,13,17,19,23,29,31,37};

	if (f%2==0){
		return 0;
	}


	for (j = 0, N = f; j < k; ++j){

		int t;
		uint64_t curBit, exp, nmo, q, one, r2;

		mont_init(N, t, curBit, exp, nmo, q, one, r2);

		if ( N < 3825123056546413051ULL ){
			for (int i = 0; i < 9; ++i){
				if (!strong_prp(base[i], N, t, curBit, exp, nmo, q, one, r2)){
					return j;
				}
			}
		}
		else if ( N <= UINT64_MAX ){
			for (int i = 0; i < 12; ++i){
				if (!strong_prp(base[i], N, t, curBit, exp, nmo, q, one, r2)){
					return j;
				}
			}
		}

		N += (uint64_t)d*2*3*5*7*11*13*17*19*23;
	}

	return j;
}


// GPU does a prp base 2 check only. It will sometimes report an AP with a base 2 probable prime.
void ReportSolution(int AP_Length,int difference,uint64_t First_Term)
{

	int i;

	/*	add each AP10+ first_term mod 1000 and that AP's length to checksum	*/
	cksum += First_Term % 1000;
	cksum += AP_Length;
	if(cksum > MAXINTV){
		cksum -= MAXINTV;
	}

	i = validate_ap26(AP_Length,difference,First_Term);

	if (i < AP_Length){

		if(boinc_is_standalone()){
			printf("Non-Solution: %d %d %" PRIu64 "\n",AP_Length,difference,First_Term);
		}

		if (val_base2_ap26(AP_Length,difference,First_Term) < AP_Length){
			// GPU really did calculate something wrong.  It's not a prp base 2 AP
			printf("Error: Computation error, found invalid AP, exiting...\n");
			fprintf(stderr,"Error: Computation error, found invalid AP\n");
			exit(EXIT_FAILURE);
		} 

		// Even though this AP is not valid, it may contain an AP that is.
		/* Check leading terms */
		ReportSolution(i,difference,First_Term);

		/* Check trailing terms */
		ReportSolution(AP_Length-(i+1),difference,First_Term+(uint64_t)(i+1)*difference*2*3*5*7*11*13*17*19*23);
		return;
	}
	else if (AP_Length >= MINIMUM_AP_LENGTH_TO_REPORT){

		if (results_file == NULL)
			results_file = my_fopen(RESULTS_FILENAME,"a");

		if(boinc_is_standalone()){
			printf("Solution: %d %d %" PRIu64 "\n",AP_Length,difference,First_Term);
		}

		if (results_file == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}

		if (fprintf(results_file,"%d %d %" PRIu64 "\n",AP_Length,difference,First_Term)<0){
			fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
	}
	
}

/* Checkpoint 
*/
void checkpoint(int SHIFT, int K, int force)
{
	double d;
	time_t curr_time;

	time(&curr_time);
	int diff = (int)curr_time - (int)last_ckpt;

	if( diff > 60 || force ){

		last_ckpt = curr_time;

		if (results_file != NULL){
			fclose(results_file);
			results_file = NULL;
		}

		write_state(KMIN,KMAX,SHIFT,K);

		if(boinc_is_standalone()){
			printf("Checkpoint: KMIN:%d KMAX:%d SHIFT:%d K:%d\n",KMIN,KMAX,SHIFT,K);
		}

		boinc_checkpoint_completed();

		handle_trickle_up();

	}

}

/* Returns 1 iff K will be searched.
 */
int will_search(int K)
{
  	return (K%PRIME1 && K%PRIME2 && K%PRIME3 && K%PRIME4 &&
          	K%PRIME5 && K%PRIME6 && K%PRIME7 && K%PRIME8);
}

// Definition of SearchAP26()
#include "AP26.h"


#ifdef _WIN32
double getSysOpType()
{
    double ret = 0.0;
    NTSTATUS(WINAPI *RtlGetVersion)(LPOSVERSIONINFOEXW);
    OSVERSIONINFOEXW osInfo;

    *(FARPROC*)&RtlGetVersion = GetProcAddress(GetModuleHandleA("ntdll"), "RtlGetVersion");

    if (NULL != RtlGetVersion)
    {
        osInfo.dwOSVersionInfoSize = sizeof(osInfo);
        RtlGetVersion(&osInfo);
        ret = (double)osInfo.dwMajorVersion;
    }
    return ret;
}
#endif


int main(int argc, char *argv[])
{
	int i, K, SHIFT, computeunits;
	int profile = 1;
	int COMPUTE = 0;

	// disable kernel cache.  compile every time, for testing.
	// linux
	// setenv("CUDA_CACHE_DISABLE", "1", 1);
	// windows
	// _putenv_s("CUDA_CACHE_DISABLE", "1");

        // Initialize BOINC
        BOINC_OPTIONS options;
        boinc_options_defaults(options);
        options.normal_thread_priority = true;    // Raise thread priority to keep GPU busy
        boinc_init_options(&options);

	fprintf(stderr, "AP26 OpenCL 10-shift search version %s by Bryan Little\n",VERS);
	fprintf(stderr, "Compiled " __DATE__ " with GCC " __VERSION__ "\n");

	if(boinc_is_standalone()){
		printf("AP26 OpenCL 10-shift search version %s by Bryan Little\n",VERS);
		printf("Compiled " __DATE__ " with GCC " __VERSION__ "\n");
	}

	// Print out cmd line for diagnostics
        fprintf(stderr, "Command line: ");
        for (i = 0; i < argc; i++)
        	fprintf(stderr, "%s ", argv[i]);
        fprintf(stderr, "\n");

	/* Get search parameters from command line */
	if(argc < 4){
		printf("Usage: %s KMIN KMAX SHIFT\n",argv[0]);
		exit(EXIT_FAILURE);
	}

	sscanf(argv[1],"%d",&KMIN);
	sscanf(argv[2],"%d",&KMAX);
	sscanf(argv[3],"%d",&SHIFT);

	/* Resume from checkpoint if there is one */
	if (read_state(KMIN,KMAX,SHIFT,&K)){
		if(boinc_is_standalone()){
			printf("Resuming search from checkpoint.\n");
		}
		fprintf(stderr,"Resuming from checkpoint. K: %d\n",K);
	}
	else{
		if(boinc_is_standalone()){
			printf("Beginning a new search with parameters from the command line\n");
		}
		K = KMIN;
		cksum = 0; // zero result checksum for BOINC
		totalaps = 0;  // total count of APs found
		write_state_a_next = true;

		// clear result file
		FILE * temp_file = my_fopen(RESULTS_FILENAME,"w");
		if (temp_file == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
		fclose(temp_file);

		// setup boinc trickle up
		last_trickle = (uint64_t)time(NULL);
	}

	//trying to resume a finished workunit
	if(K > KMAX){
		if(boinc_is_standalone()){
			printf("Workunit complete.\n");
		}
		fprintf(stderr,"Workunit complete.\n");
		boinc_finish(EXIT_SUCCESS);
		return EXIT_SUCCESS;
	}


	cl_int err;
	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context ctx;
	cl_command_queue queue;

	int retval = 0;
	retval = boinc_get_opencl_ids(argc, argv, 0, &device, &platform);
	if (retval) {
		if(boinc_is_standalone()){
			printf("init_data.xml not found, using device 0.\n");

			err = clGetPlatformIDs(1, &platform, NULL);
			if (err != CL_SUCCESS) {
				printf( "clGetPlatformIDs() failed with %d\n", err );
				exit(EXIT_FAILURE);
			}
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
			if (err != CL_SUCCESS) {
				printf( "clGetDeviceIDs() failed with %d\n", err );
				exit(EXIT_FAILURE);
			}
		}
		else{
			fprintf(stderr, "Error: boinc_get_opencl_ids() failed with error %d\n", retval );
			exit(EXIT_FAILURE);
		}
	}

	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	ctx = clCreateContext(cps, 1, &device, NULL, NULL, &err);
	if (err != CL_SUCCESS) {
		fprintf(stderr, "Error: clCreateContext() returned %d\n", err);
        	exit(EXIT_FAILURE); 
   	}

	queue = clCreateCommandQueue(ctx, device, CL_QUEUE_PROFILING_ENABLE, &err);	
	//queue = clCreateCommandQueue(ctx, device, 0, &err);
	if(err != CL_SUCCESS) { 
		fprintf(stderr, "Error: Creating Command Queue. (clCreateCommandQueue) returned %d\n", err );
		exit(EXIT_FAILURE);
    	}

	hardware.platform = platform;
	hardware.device = device;
	hardware.queue = queue;
	hardware.context = ctx;

 	char device_name[1024];
 	char device_vend[1024];
 	char device_driver[1024];
	cl_uint CUs;

	err = clGetDeviceInfo(hardware.device, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
	if ( err != CL_SUCCESS ) {
		if(boinc_is_standalone()){
			printf( "Error: clGetDeviceInfo\n" );
		}
		fprintf(stderr, "Error: clGetDeviceInfo\n" );
		exit(EXIT_FAILURE);
	}

	err = clGetDeviceInfo(hardware.device, CL_DEVICE_VENDOR, sizeof(device_vend), &device_vend, NULL);
	if ( err != CL_SUCCESS ) {
		if(boinc_is_standalone()){
			printf( "Error: clGetDeviceInfo\n" );
		}
		fprintf(stderr, "Error: clGetDeviceInfo\n" );
		exit(EXIT_FAILURE);
	}

	err = clGetDeviceInfo(hardware.device, CL_DRIVER_VERSION, sizeof(device_driver), &device_driver, NULL);
	if ( err != CL_SUCCESS ) {
		if(boinc_is_standalone()){
			printf( "Error: clGetDeviceInfo\n" );
		}
		fprintf(stderr, "Error: clGetDeviceInfo\n" );
		exit(EXIT_FAILURE);
	}

	err = clGetDeviceInfo(hardware.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &CUs, NULL);
	if ( err != CL_SUCCESS ) {
		if(boinc_is_standalone()){
			printf( "Error: clGetDeviceInfo\n" );
		}
		fprintf(stderr, "Error: clGetDeviceInfo\n" );
		exit(EXIT_FAILURE);
	}

	fprintf(stderr, "GPU Info:\n  Name: \t\t%s\n  Vendor: \t\t%s\n  Driver: \t\t%s\n  Compute Units: \t%u\n", device_name, device_vend, device_driver, CUs);
	if(boinc_is_standalone()){
		printf("GPU Info:\n  Name: \t\t%s\n  Vendor: \t\t%s\n  Driver: \t\t%s\n  Compute Units: \t%u\n", device_name, device_vend, device_driver, CUs);
	}

	// check vendor and normalize compute units. doesn't have to be accurate, work size is determined by kernel runtime.
	computeunits = (int)CUs;

	char intel_s[] = "Intel";
	char arc_s[] = "Arc";
	char nvidia_s[] = "NVIDIA";
	
	if(strstr((char*)device_vend, (char*)nvidia_s) != NULL){

	 	cl_uint ccmajor;

		err = clGetDeviceInfo(hardware.device, CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, sizeof(ccmajor), &ccmajor, NULL);
		if ( err != CL_SUCCESS ) {
			if(boinc_is_standalone()){
		        	printf( "Error: clGetDeviceInfo\n" );
			}
			fprintf(stderr, "Error: clGetDeviceInfo\n" );
		        exit(EXIT_FAILURE);
		}

		if(ccmajor < 7){
			// older nvidia gpus
		        printf("compiling sieve for NVIDIA with local mem cache\n");
		        sieve = sclGetCLSoftware(sieve_nv_cl,"sieve",hardware, 1);

			// kernel has __attribute__ ((reqd_work_group_size(1024, 1, 1)))
			// Nvidia's 4xx.x drivers changed CL_KERNEL_WORK_GROUP_SIZE return value to 256
			// this kernel runs much quicker (33%+) at 1024 because of the local memory copy
			// hack around nvidia's driver change
			if(sieve.local_size[0] != 1024){
				sieve.local_size[0] = 1024;
				fprintf(stderr, "Set sieve kernel local size to 1024\n");
				printf("Set sieve kernel local size to 1024\n");
			}

		}
		else{
			// current gpus with big L2 cache
		        printf("compiling sieve\n");
		        sieve = sclGetCLSoftware(sieve_cl,"sieve",hardware, 1);
		}


#ifdef _WIN32
		// pascal or newer gpu on windows 10,11 allows long kernel runtimes without screen refresh issues

		float winVer = (float)getSysOpType();

		if(winVer >= 10.0f && ccmajor >= 6){
			COMPUTE = 1;
		}

#else
		// linux
		// data center or mining card without video output	
		// long kernel runtimes are ok
		char dc0[] = "H100";
		char dc1[] = "A100";
		char dc2[] = "V100";
		char dc3[] = "T4";
		char dc4[] = "P106";
		char dc5[] = "P104";
		char dc6[] = "P102";
		char dc7[] = "P100";
		char dc8[] = "CMP";
		char dc9[] = "A2";
		char dc10[] = "A10";
		char dc11[] = "A16";
		char dc12[] = "A30";
		char dc13[] = "A40";

		if(	strstr((char*)device_name, (char*)dc0) != NULL
			|| strstr((char*)device_name, (char*)dc1) != NULL
			|| strstr((char*)device_name, (char*)dc2) != NULL
			|| strstr((char*)device_name, (char*)dc3) != NULL
			|| strstr((char*)device_name, (char*)dc4) != NULL
			|| strstr((char*)device_name, (char*)dc5) != NULL
			|| strstr((char*)device_name, (char*)dc6) != NULL
			|| strstr((char*)device_name, (char*)dc7) != NULL
			|| strstr((char*)device_name, (char*)dc8) != NULL
			|| strstr((char*)device_name, (char*)dc9) != NULL
			|| strstr((char*)device_name, (char*)dc10) != NULL
			|| strstr((char*)device_name, (char*)dc11) != NULL
			|| strstr((char*)device_name, (char*)dc12) != NULL
			|| strstr((char*)device_name, (char*)dc13) != NULL){
			COMPUTE = 1;
		}

#endif



	}
	// Intel
	else if( strstr((char*)device_vend, (char*)intel_s) != NULL ){

		if( strstr((char*)device_name, (char*)arc_s) != NULL ){
			computeunits /= 10;
		}
		else{
			computeunits /= 20;
	                fprintf(stderr,"Detected Intel integrated graphics\n");	
		}

                printf("compiling sieve\n");
                sieve = sclGetCLSoftware(sieve_cl,"sieve",hardware, 1);

	}
	// AMD
        else{
		computeunits /= 2;

                printf("compiling sieve\n");
                sieve = sclGetCLSoftware(sieve_cl,"sieve",hardware, 1);
        }


	if(computeunits < 1){
		computeunits = 1;
	}
	
	// build kernels
	printf("compiling clearok\n");
        clearok = sclGetCLSoftware(clearok_cl,"clearok",hardware, 1);

	printf("compiling clearokok\n");
        clearokok = sclGetCLSoftware(clearokok_cl,"clearokok",hardware, 1);

	printf("compiling clearn\n");
        clearn = sclGetCLSoftware(clearn_cl,"clearn",hardware, 1);

	printf("compiling offset\n");
        offset = sclGetCLSoftware(offset_cl,"offset",hardware, 1);

	printf("compiling setupok\n");
        setupok = sclGetCLSoftware(setupok_cl,"setupok",hardware, 1);

	printf("compiling setupn\n");
        setupn = sclGetCLSoftware(setupn_cl,"setupn",hardware, 1);

        printf("compiling setupokok\n");
        setupokok = sclGetCLSoftware(setupokok_cl,"setupokok",hardware, 1);

        printf("compiling checkn\n");
        checkn = sclGetCLSoftware(checkn_cl,"checkn",hardware, 1);

	printf("Kernel compile done.\n");


	// setup kernel global sizes
	sclSetGlobalSize( clearn, 64 );
	sclSetGlobalSize( clearokok, 23693 );
	sclSetGlobalSize( clearok, 23693 );
	sclSetGlobalSize( setupn, 10840 );
	sclSetGlobalSize( offset, 542 );
	sclSetGlobalSize( setupokok, 542 );
	sclSetGlobalSize( setupok, 542 );


        // memory allocation
        // host memory
        n43_h = (uint64_t*)malloc(numn43s * sizeof(uint64_t));
        sol_k_h = (int*)malloc(sol * sizeof(int));
        sol_val_h = (uint64_t*)malloc(sol * sizeof(uint64_t));
	counter_h = (int*)malloc(4 * sizeof(int));
        // device memory
        n43_d = sclMalloc(hardware, CL_MEM_READ_WRITE, numn43s * sizeof(uint64_t));
        n59_0_d = sclMalloc(hardware, CL_MEM_READ_WRITE, halfn59s * sizeof(uint64_t));
        n59_1_d = sclMalloc(hardware, CL_MEM_READ_WRITE, halfn59s * sizeof(uint64_t));
        counter_d = sclMalloc(hardware, CL_MEM_READ_WRITE, 4 * sizeof(int));
        OKOK_d = sclMalloc(hardware, CL_MEM_READ_WRITE, numOK * sizeof(uint64_t));
        OK_d = sclMalloc(hardware, CL_MEM_READ_WRITE, numOK * sizeof(char));
        offset_d = sclMalloc(hardware, CL_MEM_READ_WRITE, 542 * sizeof(int));
        sol_k_d = sclMalloc(hardware, CL_MEM_READ_WRITE, sol * sizeof(int));
        sol_val_d = sclMalloc(hardware, CL_MEM_READ_WRITE, sol * sizeof(uint64_t));


	/* Count the number of K in the range KMIN <= K <= KMAX that will actually
		be searched and (if K > KMIN) those that have already been searched. */
	for (i = KMIN; i <= KMAX; i++){
		if (will_search(i)){
			K_COUNT++;
			if (K > i)
				K_DONE++;
		}
	}


	time_t totals, totalf;
	if(boinc_is_standalone()){
		time(&totals);
	}

	time(&last_ckpt);

	/* Top-level loop */
	for (; K <= KMAX; ++K){
		if (will_search(K)){

			checkpoint(SHIFT,K,0);

			SearchAP26(K,SHIFT,profile,computeunits,COMPUTE);

		 	K_DONE++;

			Progress((double)K_DONE / (double)K_COUNT);
		}
	}

	if(boinc_is_standalone()){
		time(&totalf);
		printf("search finished in %d sec.\n", (int)totalf - (int)totals);
	}


	boinc_begin_critical_section();
	boinc_fraction_done(1.0);
	checkpoint(SHIFT,K,1);
	write_cksum();
	fprintf(stderr,"Workunit complete.  Number of AP10+ found %u\n", totalaps);
	boinc_end_critical_section();

        // free memory
        // host
        free(n43_h);
        free(sol_k_h);
        free(sol_val_h);
	free(counter_h);

        // device
        sclReleaseMemObject(counter_d);
        sclReleaseMemObject(n43_d);
        sclReleaseMemObject(n59_0_d);
        sclReleaseMemObject(n59_1_d);
        sclReleaseMemObject(OK_d);
        sclReleaseMemObject(OKOK_d);
        sclReleaseMemObject(offset_d);
        sclReleaseMemObject(sol_k_d);
        sclReleaseMemObject(sol_val_d);
        sclReleaseMemObject(n_result_d);

        //free scl
        sclReleaseClSoft(clearok);
        sclReleaseClSoft(clearokok);
        sclReleaseClSoft(clearn);
        sclReleaseClSoft(offset);
        sclReleaseClSoft(checkn);
        sclReleaseClSoft(setupokok);
        sclReleaseClSoft(setupok);
        sclReleaseClSoft(sieve);
        sclReleaseClSoft(setupn);

        sclReleaseClHard(hardware);

	boinc_finish(EXIT_SUCCESS);



  return EXIT_SUCCESS;
}
