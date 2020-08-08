/* 
	AP26 Multithreaded CPU application
 	Bryan Little
 	March 22, 2020				*/


// AP26 application version
#define MAJORV 3
#define MINORV 0
#define SUFFIXV ""

#include <iostream>
#include <cinttypes>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <algorithm>

#include <pthread.h>
#include <thread>

#include "boinc_api.h"
#include "filesys.h"

#include "mainconst.h"
#include "prime.h"

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
int result_hash;
bool write_state_a_next;
static FILE *results_file = NULL;
int64_t *n43_h;


/////////////////////////////
// main lock for search data
uint32_t current_n43;
pthread_mutex_t lock1;
/////////////////////////////


///////////////////////////////////
// lock used for reporting results
pthread_mutex_t lock2;
///////////////////////////////////


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


// Bryan Little 6-9-2016
// BOINC result hash calculation, write to solution file, and close.
// The hash function is a 16 char hexadecimal string used to compare results found by different computers in a BOINC quorum.
// The hash can be used to compare results between GPU and CPU clients.
// It also prevents the server from having to validate every AP10+ reported by clients, which can be in different orders depending on GPU.
// It contains information about the assigned workunit and all APs found of length 10 or larger.
static void write_hash()
{
	// calculate the top 32bits of the hash based on assigned workunit range
	uint64_t minmax = KMIN + KMAX;
	// check to make sure we don't overflow a 32bit signed int with large K values
	while(minmax > MAXINTV){
		minmax -= MAXINTV;
	}

	// top 32 bits are workunit range... bottom 32 bits are solutions found

	int64_t hash = (int64_t)( (minmax << 32) | result_hash);

	if (results_file == NULL)
		results_file = my_fopen(RESULTS_FILENAME,"a");

	if (results_file == NULL){
		fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
		exit(EXIT_FAILURE);
	}

	if (fprintf(results_file,"%016" PRIX64 "\n",hash)<0){
		fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
		exit(EXIT_FAILURE);
	}

	fclose(results_file);

}


static void write_state(int KMIN, int KMAX, int SHIFT, int K)
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

	if (fprintf(out,"%d %d %d %d %d\n",KMIN,KMAX,SHIFT,K,result_hash) < 0){
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
static int read_state(int KMIN, int KMAX, int SHIFT, int *K)
{
	FILE *in;
	bool good_state_a = true;
	bool good_state_b = true;
	int tmp1, tmp2, tmp3;
	int K_a, hash_a, K_b, hash_b;

        // Attempt to read state file A
	if ((in = my_fopen(STATE_FILENAME_A,"r")) == NULL)
        {
		good_state_a = false;
        }
	else if (fscanf(in,"%d %d %d %d %d\n",&tmp1,&tmp2,&tmp3,&K_a,&hash_a) != 5)
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
        else if (fscanf(in,"%d %d %d %d %d\n",&tmp1,&tmp2,&tmp3,&K_b,&hash_b) != 5)
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
		result_hash = hash_a;
		write_state_a_next = false;
		return 1;
	}
        if (good_state_b && !good_state_a)
        {
                *K = K_b;
                result_hash = hash_b;
		write_state_a_next = true;
		return 1;
        }

	// If we got here, neither state file was good
	return 0;
}

/* Bryan Little 9-28-2015
   Bryan Little - added to CPU code 6-9-2016
   Returns index j where:
   0<=j<k ==> f+j*d*23# is composite.
   j=k    ==> for all 0<=j<k, f+j*d*23# is a strong probable prime to base 2 only.
*/
static int val_base2_ap26(int k, int d, int64_t f)
{
	int64_t N;
	int j;

	if (f%2==0)
		return 0;

	for (j = 0, N = f; j < k; j++){

		if (!strong_prp(2,N))
			return j;

		N += (int64_t)d*2*3*5*7*11*13*17*19*23;
	}

	return j;
}

/*   Bryan Little - added to CPU code 6-9-2016
   Returns index j where:
   0<=j<k ==> f+j*d*23# is composite.
   j=k    ==> for all 0<=j<k, f+j*d*23# is a strong probable prime to 9 bases.
*/
static int validate_ap26(int k, int d, int64_t f)
{
  int64_t N;
  int j;

  const int base[] = {2,3,5,7,11,13,17,19,23};

  if (f%2==0)
    return 0;


  for (j = 0, N = f; j < k; j++)
  {

    int i;

    for (i = 0; i < sizeof(base)/sizeof(int); i++)
      if (!strong_prp(base[i],N))
        return j;

    N += (int64_t)d*2*3*5*7*11*13*17*19*23;
  }

  return j;
}

// Bryan Little 9-28-2015
// Bryan Little - added to CPU code 6-9-2016
// Changed function to check ALL solutions for validity, not just solutions >= MINIMUM_AP_LENGTH_TO_REPORT
// CPU does a prp base 2 check only. It will sometimes report an AP with a base 2 probable prime.
void ReportSolution(int AP_Length,int difference,int64_t First_Term)
{

	int i;

	/* 	hash for BOINC quorum
		we create a hash based on each AP10+ result mod 1000
		and that result's AP length  	*/
	result_hash += First_Term % 1000;
	result_hash += AP_Length;
	if(result_hash > MAXINTV){
		result_hash -= MAXINTV;
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

	if (force || boinc_time_to_checkpoint()){

		if (results_file != NULL){
			fflush(results_file);
#if defined (_WIN32)
			_commit(_fileno(results_file));
#else
			fsync(fileno(results_file));
#endif
                }

		write_state(KMIN,KMAX,SHIFT,K);

		if(boinc_is_standalone()){
			printf("Checkpoint: KMIN:%d KMAX:%d SHIFT:%d K:%d\n",KMIN,KMAX,SHIFT,K);
		}

		if (!force)
			boinc_checkpoint_completed();
	}

	if(force){
		if (K_COUNT > 0)
			d = (double)(K_DONE / K_COUNT);
		else
			d = 1.0;

		boinc_fraction_done(d);
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

	// initialize pthread mutex
	err = pthread_mutex_init(&lock1, NULL);
	if (err){
		fprintf(stderr, "ERROR: pthread_mutex_init, code: %d\n", err);
		exit(EXIT_FAILURE);
	}
	err = pthread_mutex_init(&lock2, NULL);
	if (err){
		fprintf(stderr, "ERROR: pthread_mutex_init, code: %d\n", err);
		exit(EXIT_FAILURE);
	}

	fprintf(stderr, "AP26 CPU 10-shift search version %d.%d%s by Bryan Little and Iain Bethune\n",MAJORV,MINORV,SUFFIXV);
	fprintf(stderr, "Compiled " __DATE__ " with GCC " __VERSION__ "\n");

	if(boinc_is_standalone()){
		printf("AP26 CPU 10-shift search version %d.%d%s by Bryan Little and Iain Bethune\n",MAJORV,MINORV,SUFFIXV);
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
	int avx512 = __builtin_cpu_supports("avx512dq");

	if(avx512){
		if(boinc_is_standalone()){
			printf("Detected avx512dq CPU\n");
		}
		fprintf(stderr, "Detected avx512dq CPU\n");
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
					printf("forcing avx512dq mode\n");
				}
				fprintf(stderr, "forcing avx512dq mode\n");
				if(avx512 == 0){
					if(boinc_is_standalone()){
						printf("ERROR: CPU does not support avx512dq instructions!\n");
					}
					fprintf(stderr, "ERROR: CPU does not support avx512dq instructions!\n");
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
		result_hash = 0; // zero result hash for BOINC
                write_state_a_next = true;

		// clear result file
		FILE * temp_file = my_fopen(RESULTS_FILENAME,"w");
		if (temp_file == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
		fclose(temp_file);
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


        n43_h = (int64_t*)malloc(numn43s * sizeof(int64_t));


	/* Top-level loop */
	for (; K <= KMAX; ++K){
		if (will_search(K)){

			if(boinc_is_standalone()){
				checkpoint(SHIFT,K,1);
			}
			else{
				checkpoint(SHIFT,K,0);
			}

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
	/* Force Final checkpoint */
	checkpoint(SHIFT,K,1);
	/* Write BOINC hash to file */
	write_hash();
	fprintf(stderr,"Workunit complete.\n");
	boinc_end_critical_section();

	free(n43_h);

	boinc_finish(EXIT_SUCCESS);
	return EXIT_SUCCESS;
}
