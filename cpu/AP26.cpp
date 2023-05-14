/* 
	AP26 Multithreaded CPU application
 	Bryan Little
 	May 13, 2023				*/


#include <cinttypes>
#include <cstdio>
#include <pthread.h>
#include <thread>

#include "boinc_api.h"
#include "filesys.h"

#include "mainconst.h"

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

// a bit less than 32bit signed int max
#define MAXINTV 2000000000

#define STATE_FILENAME_A "AP26-state.a.txt"
#define STATE_FILENAME_B "AP26-state.b.txt"
#define RESULTS_FILENAME "SOL-AP26.txt"

#define MINIMUM_AP_LENGTH_TO_REPORT 20

using namespace std; 

/* Global variables */
static int KMIN, KMAX, K_DONE, K_COUNT;
static FILE *results_file = NULL;
uint64_t *n43_h;
bool write_state_a_next;
uint32_t cksum;
uint64_t last_trickle;
time_t last_ckpt;


/////////////////////////////
// main lock for search data
int current_n43;
pthread_mutex_t lock1;
/////////////////////////////


///////////////////////////////////
// lock used for reporting results
uint32_t totalaps;
pthread_mutex_t lock2;
///////////////////////////////////


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


// strong probable prime to base 2
bool PrimeQ(uint64_t N)
{
	uint64_t nmo = N-1;
	int t = __builtin_ctzll(nmo);
	uint64_t exp = N >> t;
	uint64_t curBit = 0x8000000000000000;
	curBit >>= ( __builtin_clzll(exp) + 1 );
	uint64_t q = invert(N);
	uint64_t one = (-N) % N;
	nmo = N - one;
	uint64_t two = add(one, one, N);
	
	uint64_t a = two;

	/* If N is prime and N = d*2^t+1, where d is odd, then either
		1.  a^d = 1 (mod N), or
		2.  a^(d*2^s) = -1 (mod N) for some s in 0 <= s < t    */

  	/* r <-- a^d mod N, assuming d odd */
	while( curBit )
	{
		a = montMul(a,a,N,q);

		if(exp & curBit){
			a = add(a,a,N);
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


void ckerr(int err){
	if(err){
		fprintf(stderr, "ERROR: pthreads, code: %d\n", err);
		exit(EXIT_FAILURE);
	}
}

int boinc_standalone()
{
	return boinc_is_standalone();
}


static FILE *my_fopen(const char *filename, const char *mode)
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


// Bryan Little 9-28-2015
// Bryan Little - added to CPU code 6-9-2016
// Changed function to check ALL solutions for validity, not just solutions >= MINIMUM_AP_LENGTH_TO_REPORT
// CPU does a prp base 2 check only. It will sometimes report an AP with a base 2 probable prime.
void ReportSolution(int AP_Length, int difference, uint64_t First_Term)
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
			printf("Non-Solution: %d %d %" PRId64 "\n",AP_Length,difference,First_Term);
		}

		if (val_base2_ap26(AP_Length,difference,First_Term) < AP_Length){
			// CPU really did calculate something wrong.  It's not a prp base 2 AP
			printf("Error: Computation error, found invalid AP, exiting...\n");
			fprintf(stderr,"Error: Computation error, found invalid AP\n");
			exit(EXIT_FAILURE);
		} 

		// Even though this AP is not valid, it may contain an AP that is.
		/* Check leading terms */
		ReportSolution(i,difference,First_Term);

		/* Check trailing terms */
		ReportSolution(AP_Length-(i+1),difference,First_Term+(int64_t)(i+1)*difference*2*3*5*7*11*13*17*19*23);
		return;
	}
	else if (AP_Length >= MINIMUM_AP_LENGTH_TO_REPORT){

		if (results_file == NULL)
			results_file = my_fopen(RESULTS_FILENAME,"a");

		if(boinc_is_standalone()){
			printf("Solution: %d %d %" PRId64 "\n",AP_Length,difference,First_Term);
		}

		if (results_file == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}

		if (fprintf(results_file,"%d %d %" PRId64 "\n",AP_Length,difference,First_Term)<0){
			fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
	}
	
}

/* Checkpoint 
   If force is nonzero then don't ask BOINC for permission.
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
static int will_search(int K)
{
  	return (K%PRIME1 && K%PRIME2 && K%PRIME3 && K%PRIME4 &&
          	K%PRIME5 && K%PRIME6 && K%PRIME7 && K%PRIME8);
}


int main(int argc, char *argv[])
{
	int i, K, SHIFT, err;
	int num_threads = 1;

	// Initialize BOINC
	BOINC_OPTIONS options;
	boinc_options_defaults(options);
	options.multi_thread = true; 
	boinc_init_options(&options);
		
	n43_h = (uint64_t*)malloc(numn43s * sizeof(uint64_t));		

	ckerr(pthread_mutex_init(&lock1, NULL));
	ckerr(pthread_mutex_init(&lock2, NULL));

	fprintf(stderr, "AP26 CPU 10-shift search version %s by Bryan Little\n",VERS);
	fprintf(stderr, "Compiled " __DATE__ " with GCC " __VERSION__ "\n");

	if(boinc_is_standalone()){
		printf("AP26 CPU 10-shift search version %s by Bryan Little\n",VERS);
		printf("Compiled " __DATE__ " with GCC " __VERSION__ "\n");
	}

	// Print out cmd line for diagnostics
	fprintf(stderr, "Command line: ");
	for (i = 0; i < argc; i++)
		fprintf(stderr, "%s ", argv[i]);
	fprintf(stderr, "\n");


	/* Get search parameters from command line */
	if(argc < 4){
		printf("Usage: %s KMIN KMAX SHIFT -cputype -t #\n",argv[0]);
		printf("-cputype is used to force an instruction set. Valid types: -sse2 -sse41 -avx -avx2 -avx512. Default is highest available.\n");
		printf("-t # or --nthreads # is optional number of threads to use. Default is 1. Max is 64.\n");

		exit(EXIT_FAILURE);
	}

	sscanf(argv[1],"%d",&KMIN);
	sscanf(argv[2],"%d",&KMAX);
	sscanf(argv[3],"%d",&SHIFT);

	int sse41 = __builtin_cpu_supports("sse4.1");
	int avx = __builtin_cpu_supports("avx");
	int avx2 = __builtin_cpu_supports("avx2");
	int avx512 = __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl");

	if(avx512){
		if(boinc_is_standalone()){
			printf("Detected avx512 CPU\n");
		}
		fprintf(stderr, "Detected avx512 CPU\n");
	}
	else if(avx2){
		if(boinc_is_standalone()){
			printf("Detected avx2 CPU\n");
		}
		fprintf(stderr, "Detected avx2 CPU\n");
	}
	else if(avx){
		if(boinc_is_standalone()){
			printf("Detected avx CPU\n");
		}
		fprintf(stderr, "Detected avx CPU\n");
	}
	else if(sse41){
		if(boinc_is_standalone()){
			printf("Detected sse4.1 CPU\n");
		}
		fprintf(stderr, "Detected sse4.1 CPU\n");		
	}
	else{
		if(boinc_is_standalone()){
			printf("Assumed sse2 CPU\n");
		}
		fprintf(stderr, "Assumed sse2 CPU\n");
	}

	if(argc > 4){
		for(int xv=4;xv<argc;xv++){
			if( ( strcmp(argv[xv], "-t") == 0 || strcmp(argv[xv], "--nthreads") == 0 ) && xv+1 < argc){

				uint32_t NT;
				sscanf(argv[xv+1],"%u",&NT);

				if(NT < 1){
					if(boinc_is_standalone()){
						printf("ERROR: number of threads must be at least 1.\n");
					}
					fprintf(stderr, "ERROR: number of threads must be at least 1.\n");
					exit(EXIT_FAILURE);
				}
				else if(NT > 64){
					NT=64;
					if(boinc_is_standalone()){
						printf("maximum value for number of threads is 64.\n");
					}
					fprintf(stderr, "maximum value for number of threads is 64.\n");
				}

				uint32_t maxthreads = 0;
				maxthreads = std::thread::hardware_concurrency();

				if(maxthreads){
					if(NT > maxthreads){
						if(boinc_is_standalone()){
							printf("Detected %u logical processors.  Using %u threads.\n", maxthreads, maxthreads);
						}
						fprintf(stderr, "Detected %u logical processors.  Using %u threads.\n", maxthreads, maxthreads);
						NT = maxthreads;
					}
					else{
						if(boinc_is_standalone()){
							printf("Detected %u logical processors.  Using %u threads.\n", maxthreads, NT);
						}
						fprintf(stderr, "Detected %u logical processors.  Using %u threads.\n", maxthreads, NT);
					}
				}
				else{
					if(boinc_is_standalone()){
						printf("Unable to detect logical processor count.  Using %u threads.\n", NT);
					}
					fprintf(stderr, "Unable to detect logical processor count.  Using %u threads.\n", NT);
				}
					
				num_threads = NT;
			}
			else if( strcmp(argv[xv], "-sse2") == 0 ){
				if(boinc_is_standalone()){
					printf("forcing sse2 mode\n");
				}
				fprintf(stderr, "forcing sse2 mode\n");
				sse41 = 0;
				avx = 0;
				avx2 = 0;
				avx512 = 0;
			}
			else if( strcmp(argv[xv], "-sse41") == 0 ){
				if(boinc_is_standalone()){
					printf("forcing sse4.1 mode\n");
				}
				fprintf(stderr, "forcing sse4.1 mode\n");
				if(sse41 == 0){
					if(boinc_is_standalone()){
						printf("ERROR: CPU does not support SSE4.1 instructions!\n");
					}
					fprintf(stderr, "ERROR: CPU does not support SSE4.1 instructions!\n");
					exit(EXIT_FAILURE);
				}
				sse41 = 1;
				avx = 0;
				avx2 = 0;
				avx512 = 0;
			}
			else if( strcmp(argv[xv], "-avx") == 0 ){
				if(boinc_is_standalone()){
					printf("forcing avx mode\n");
				}
				fprintf(stderr, "forcing avx mode\n");
				if(avx == 0){
					if(boinc_is_standalone()){
						printf("ERROR: CPU does not support avx instructions!\n");
					}
					fprintf(stderr, "ERROR: CPU does not support avx instructions!\n");
					exit(EXIT_FAILURE);
				}
				sse41 = 0;
				avx = 1;
				avx2 = 0;
				avx512 = 0;
			}
			else if( strcmp(argv[xv], "-avx2") == 0 ){
				if(boinc_is_standalone()){
					printf("forcing avx2 mode\n");
				}
				fprintf(stderr, "forcing avx2 mode\n");
				if(avx2 == 0){
					if(boinc_is_standalone()){
						printf("ERROR: CPU does not support avx2 instructions!\n");
					}
					fprintf(stderr, "ERROR: CPU does not support avx2 instructions!\n");
					exit(EXIT_FAILURE);
				}
				sse41 = 0;
				avx = 0;
				avx2 = 1;
				avx512 = 0;
			}
			else if( strcmp(argv[xv], "-avx512") == 0 ){
				if(boinc_is_standalone()){
					printf("forcing avx512 mode\n");
				}
				fprintf(stderr, "forcing avx512 mode\n");
				if(avx512 == 0){
					if(boinc_is_standalone()){
						printf("ERROR: CPU does not support avx512 instructions!\n");
					}
					fprintf(stderr, "ERROR: CPU does not support avx512 instructions!\n");
					exit(EXIT_FAILURE);
				}
				sse41 = 0;
				avx = 0;
				avx2 = 0;
				avx512 = 1;
			}
		}
	}


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
		boinc_finish(EXIT_SUCCESS);
		return EXIT_SUCCESS;
	}


	/* Count the number of K in the range KMIN <= K <= KMAX that will actually
		be searched and (if K > KMIN) those that have already been searched. */
	for (i = KMIN; i <= KMAX; i++){
		if (will_search(i)){
			K_COUNT++;
			if (K > i)
				K_DONE++;
		}
	}


	/* Top-level loop */
	for (; K <= KMAX; ++K){
		if (will_search(K)){

			checkpoint(SHIFT,K,0);

			if(avx512){
				Search_avx512(K, SHIFT, K_COUNT, K_DONE, num_threads);
			}
			else if(avx2){
				Search_avx2(K, SHIFT, K_COUNT, K_DONE, num_threads);
			}
			else if(avx){
				Search_avx(K, SHIFT, K_COUNT, K_DONE, num_threads);
			}
			else if(sse41){
				Search_sse41(K, SHIFT, K_COUNT, K_DONE, num_threads);
			}
			else{
				Search_sse2(K, SHIFT, K_COUNT, K_DONE, num_threads);
			}

		 	K_DONE++;
		}
	}



	boinc_begin_critical_section();
	boinc_fraction_done(1.0);
	checkpoint(SHIFT,K,1);
	write_cksum();
	fprintf(stderr,"Workunit complete.  Number of AP10+ found %u\n", totalaps);
	boinc_end_critical_section();

	free(n43_h);

	boinc_finish(EXIT_SUCCESS);
	return EXIT_SUCCESS;
}
