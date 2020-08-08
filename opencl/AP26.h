
/*
	***************
	Bryan Little March 20, 2020

*/


// sleep CPU thread while GPU is busy
void sleepcpu(int K){

	cl_event kernelsDone;
	cl_int err;
	cl_int info;
	struct timespec sleep_time;
	sleep_time.tv_sec = 0;
	sleep_time.tv_nsec = 1000000;	// 1ms

	err = clEnqueueMarker( hardware.queue, &kernelsDone);
	if ( err != CL_SUCCESS ) {
		printf( "Error: clEnqueueMarker");
		fprintf(stderr, "Error: clEnqueueMarker, possible kernel crash at K: %d\n", K);
		sclPrintErrorFlags(err); 
	}

	clFlush(hardware.queue);

	err = clGetEventInfo(kernelsDone, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &info, NULL);
	if ( err != CL_SUCCESS ) {
		printf( "Error: clGetEventInfo\n" );
		fprintf(stderr, "Error: clGetEventInfo, possible kernel crash at K: %d\n", K);
		sclPrintErrorFlags( err );
       	}

	// sleep until event complete
	while(info >= 0 && info != CL_COMPLETE){
		nanosleep(&sleep_time,NULL);
		err = clGetEventInfo(kernelsDone, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &info, NULL);
		if ( err != CL_SUCCESS ) {
			printf( "Error: clGetEventInfo\n" );
			fprintf(stderr, "Error: clGetEventInfo, possible kernel crash at K: %d\n", K);
			sclPrintErrorFlags( err );
	       	}
	}
	clReleaseEvent(kernelsDone);
}


void SearchAP26(int K, int startSHIFT, int & profile, uint32_t CU)
{ 

	int64_t STEP;
	int64_t n0;

	int i3, i5, i31, i37, i41;
	int64_t S31, S37, S41, S43, S47, S53, S59;
	int numinq=0;
	double d = (double)1.0 / (K_COUNT*10);
	double dd;
	int SHIFT=startSHIFT;
//	uint64_t totaln = 0;

	time_t total_start_time, total_finish_time;
	time_t last_time, curr_time;

	time (&total_start_time);

	STEP=K*PRIM23;
	n0=(N0*(K%17835)+((N0*17835)%MOD)*(K/17835)+N30)%MOD;
	S31=(PRES2*(K%17835)+((PRES2*17835)%MOD)*(K/17835))%MOD;
	S37=(PRES3*(K%17835)+((PRES3*17835)%MOD)*(K/17835))%MOD;
	S41=(PRES4*(K%17835)+((PRES4*17835)%MOD)*(K/17835))%MOD;
	S43=(PRES5*(K%17835)+((PRES5*17835)%MOD)*(K/17835))%MOD;
	S47=(PRES6*(K%17835)+((PRES6*17835)%MOD)*(K/17835))%MOD;
	S53=(PRES7*(K%17835)+((PRES7*17835)%MOD)*(K/17835))%MOD;
	S59=(PRES8*(K%17835)+((PRES8*17835)%MOD)*(K/17835))%MOD;

	// note: n59MAX == 4808136200
	// note: n53MAX == 137375320

	int count=0;

	for(i31=0;i31<7;++i31)
	for(i37=0;i37<13;++i37)
	if(i37-i31<=10&&i31-i37<=4)
	for(i41=0;i41<17;++i41)
	if(i41-i31<=14&&i41-i37<=14&&i31-i41<=4&&i37-i41<=10)
	for(i3=0;i3<2;++i3)
	for(i5=0;i5<4;++i5){ 
		n43_h[count]=(n0+i3*S3+i5*S5+i31*S31+i37*S37+i41*S41)%MOD;  //10840 of these  12673 n53 per
		count++;
	}

	// offload to gpu, blocking
	sclWrite(hardware, numn43s * sizeof(int64_t), n43_d, n43_h);

	// setup n59s kernel
	sclSetKernelArg(setupn, 0, sizeof(cl_mem), &n43_d);
	sclSetKernelArg(setupn, 1, sizeof(cl_mem), &n59_0_d);
	sclSetKernelArg(setupn, 2, sizeof(cl_mem), &n59_1_d);
	sclSetKernelArg(setupn, 3, sizeof(int64_t), &S53);
	sclSetKernelArg(setupn, 4, sizeof(int64_t), &S47);
	sclSetKernelArg(setupn, 5, sizeof(int64_t), &S43);
	sclEnqueueKernel(hardware, setupn);
	// end setup n59s

	// offset kernel
	sclSetKernelArg(offset, 0, sizeof(cl_mem), &offset_d);
	sclEnqueueKernel(hardware, offset);
	// end offset

	// clearok kernel
	sclSetKernelArg(clearok, 0, sizeof(cl_mem), &OK_d);
	sclSetKernelArg(clearok, 1, sizeof(cl_mem), &counter_d);
	sclEnqueueKernel(hardware, clearok);
	// end clearok

	// setupok kernel
	sclSetKernelArg(setupok, 0, sizeof(int64_t), &STEP);
	sclSetKernelArg(setupok, 1, sizeof(cl_mem), &OK_d);
	sclSetKernelArg(setupok, 2, sizeof(cl_mem), &offset_d);
	sclEnqueueKernel(hardware, setupok);
	// end setupok

	// profile gpu sieve kernel time, once at program start
	if(profile){
		double total_ms = 0.0;
		double kernel_ms = 0.0;
		profile = 0;

		// calculate approximate chunk size based on gpu's CU
		uint64_t multiplier = 200000;
		uint64_t worksize = CU * multiplier;
		if(worksize > halfn59s){
			worksize = halfn59s;
		}

		sclSetGlobalSize( sieve, worksize );

		uint64_t estimated = sieve.global_size[0];

		// set n result array size
		numn = sieve.global_size[0] / 2;

		// allocate
		n_result_d = sclMalloc(hardware, CL_MEM_READ_WRITE, numn * sizeof(int64_t));

		// clearokok kernel
		sclSetKernelArg(clearokok, 0, sizeof(cl_mem), &OKOK_d);
		sclEnqueueKernel(hardware, clearokok);
		// end clearokok

		// setupokok kernel
		sclSetKernelArg(setupokok, 0, sizeof(int), &SHIFT);
		sclSetKernelArg(setupokok, 1, sizeof(cl_mem), &OK_d);
		sclSetKernelArg(setupokok, 2, sizeof(cl_mem), &OKOK_d);
		sclSetKernelArg(setupokok, 3, sizeof(cl_mem), &offset_d);
		sclEnqueueKernel(hardware, setupokok);
		// end setupokok

		//set static kernel args
		sclSetKernelArg(clearn, 0, sizeof(cl_mem), &counter_d);

		int p=0;
		sclSetKernelArg(sieve, 0, sizeof(cl_mem), &n59_0_d);
		sclSetKernelArg(sieve, 1, sizeof(int64_t), &S59);
		sclSetKernelArg(sieve, 2, sizeof(int), &SHIFT);
		sclSetKernelArg(sieve, 3, sizeof(cl_mem), &n_result_d);
		sclSetKernelArg(sieve, 4, sizeof(cl_mem), &OKOK_d);
		sclSetKernelArg(sieve, 5, sizeof(cl_mem), &counter_d);
		sclSetKernelArg(sieve, 6, sizeof(int), &p);

		// spin up some kernels while profiling
		for(int w = 0; w < 6; w++){

			sclEnqueueKernel(hardware, clearn);

			kernel_ms = ProfilesclEnqueueKernel(hardware, sieve);

			if(w > 1){
				total_ms += kernel_ms;
			}
		}

		// avg of the 4 profiles
		double prof_avg_ms = total_ms / 4.0;
		if(prof_avg_ms == 0.0){
			prof_avg_ms = 1.0;
		}

		// target kernel time is 33ms
		double prof_multi = 33.0 / prof_avg_ms;

		// update chunk size based on the profile
		uint64_t new_range = (uint64_t)((double)sieve.global_size[0] * prof_multi);
		if(new_range > halfn59s){
			new_range = halfn59s;
		}

		sclSetGlobalSize( sieve, new_range );

		if(boinc_is_standalone()){
			printf("Estimated worksize: %" PRIu64 " profile: %0.3fms actual worksize: %" PRIu64 "\n", estimated, prof_avg_ms, sieve.global_size[0]);
		}
		fprintf(stderr, "Estimated worksize: %" PRIu64 " profile: %0.3fms actual worksize: %" PRIu64 "\n", estimated, prof_avg_ms, sieve.global_size[0]);

		// adjust n result array size
		sclReleaseMemObject(n_result_d);
		numn = sieve.global_size[0] / 2;
		n_result_d = sclMalloc(hardware, CL_MEM_READ_WRITE, numn * sizeof(int64_t));

		sclSetGlobalSize( checkn, numn );

	}


	// set static kernel args
	sclSetKernelArg(clearokok, 0, sizeof(cl_mem), &OKOK_d);

	sclSetKernelArg(setupokok, 1, sizeof(cl_mem), &OK_d);
	sclSetKernelArg(setupokok, 2, sizeof(cl_mem), &OKOK_d);
	sclSetKernelArg(setupokok, 3, sizeof(cl_mem), &offset_d);

	sclSetKernelArg(clearn, 0, sizeof(cl_mem), &counter_d);

	sclSetKernelArg(sieve, 1, sizeof(int64_t), &S59);
	sclSetKernelArg(sieve, 3, sizeof(cl_mem), &n_result_d);
	sclSetKernelArg(sieve, 4, sizeof(cl_mem), &OKOK_d);
	sclSetKernelArg(sieve, 5, sizeof(cl_mem), &counter_d);

	sclSetKernelArg(checkn, 0, sizeof(cl_mem), &n_result_d);
	sclSetKernelArg(checkn, 1, sizeof(int64_t), &STEP);
	sclSetKernelArg(checkn, 2, sizeof(cl_mem), &sol_k_d);
	sclSetKernelArg(checkn, 3, sizeof(cl_mem), &sol_val_d);
	sclSetKernelArg(checkn, 4, sizeof(cl_mem), &counter_d);

	time (&last_time);

	uint32_t iteration = 0;

	for(; SHIFT<(startSHIFT+640); SHIFT+=64){

		sclEnqueueKernel(hardware, clearokok);

		sclSetKernelArg(setupokok, 0, sizeof(int), &SHIFT);
		sclEnqueueKernel(hardware, setupokok);

		for(int devicearray=0; devicearray<2; devicearray++){
			for(int p=0; p<halfn59s; p+=sieve.global_size[0] ){

				// update BOINC progress every 2 sec
				time (&curr_time);
				if( ((int)curr_time - (int)last_time) > 1 ){
		    			dd = (double)(K_DONE*10+iteration) * d;
					Progress(dd);
					last_time = curr_time;
				}

				sclEnqueueKernel(hardware, clearn);

				if(devicearray == 0){
					sclSetKernelArg(sieve, 0, sizeof(cl_mem), &n59_0_d);
				}
				else if(devicearray == 1){
					sclSetKernelArg(sieve, 0, sizeof(cl_mem), &n59_1_d);
				}
				sclSetKernelArg(sieve, 2, sizeof(int), &SHIFT);
				sclSetKernelArg(sieve, 6, sizeof(int), &p);
				sclEnqueueKernel(hardware, sieve);

/*				int* numbern = (int*)malloc(3 * sizeof(int));
				sclRead(hardware, 3 * sizeof(int), counter_d, numbern);
				totaln += (int64_t)numbern[0];
				printf("K: %d narray size: %d of max %d\n",K,numbern[0],numn);
				free(numbern);
*/

				sclEnqueueKernel(hardware, checkn);

				numinq++;
				if(numinq == 3){
					// sleep CPU thread while GPU is busy every ~100ms
					sleepcpu(K);
					numinq = 0;
				}

			}

		}

		++iteration;

	}


	// sleep CPU thread while GPU is busy
	sleepcpu(K);

	// copy solutions to host memory
	// blocking read
	sclRead(hardware, 3 * sizeof(int), counter_d, counter_h);
	sclRead(hardware, sol * sizeof(int), sol_k_d, sol_k_h);
	sclRead(hardware, sol * sizeof(int64_t), sol_val_d, sol_val_h);

	//printf("largest ncount: %d / %d, solution count: %d / %d\n",counter_h[1], numn ,counter_h[2], sol);

	// check if number of candidates overflowed the array
	if(counter_h[1] > numn){
		printf("Error: checkn array overflow.\n");
		fprintf(stderr, "Error: checkn array overflow.\n");
		exit(EXIT_FAILURE);
	}
	// check if number of solutions overflowed the array
	if(counter_h[2] > sol){
		printf("Error: solution array overflow.\n");
		fprintf(stderr, "Error: solution array overflow.\n");
		exit(EXIT_FAILURE);
	}

	// report solutions
	for(int e=0; e < counter_h[2]; ++e){
		ReportSolution(sol_k_h[e],K,sol_val_h[e]);
	}

	if(boinc_is_standalone()){
		time(&total_finish_time);
		printf("K %d done in %d sec.\n", K, (int)total_finish_time - (int)total_start_time);
	}

//	printf("total n for K: %" PRIu64 "\n",totaln);  // for K 366384 this should be 38838420


}
