/* cpuavx.cpp --

*/

#include <x86intrin.h>
#include <cinttypes>
#include <cstdio>
#include <pthread.h>

#include "cpuconst.h"


__m128i ixOKOK61[61];
__m128i ixOKOK67[67];
__m128i ixOKOK71[71];
__m128i ixOKOK73[73];
__m128i ixOKOK79[79];
__m128i ixOKOK83[83];
__m128i ixOKOK89[89];
__m128i ixOKOK97[97];
__m128i ixOKOK101[101];
__m128i ixOKOK103[103];
__m128i ixOKOK107[107];
__m128i ixOKOK109[109];
__m128i ixOKOK113[113];
__m128i ixOKOK127[127];
__m128i ixOKOK131[131];
__m128i ixOKOK137[137];
__m128i ixOKOK139[139];
__m128i ixOKOK149[149];
__m128i ixOKOK151[151];
__m128i ixOKOK157[157];
__m128i ixOKOK163[163];
__m128i ixOKOK167[167];
__m128i ixOKOK173[173];
__m128i ixOKOK179[179];
__m128i ixOKOK181[181];
__m128i ixOKOK191[191];
__m128i ixOKOK193[193];
__m128i ixOKOK197[197];
__m128i ixOKOK199[199];
__m128i ixOKOK211[211];
__m128i ixOKOK223[223];
__m128i ixOKOK227[227];
__m128i ixOKOK229[229];
__m128i ixOKOK233[233];
__m128i ixOKOK239[239];
__m128i ixOKOK241[241];
__m128i ixOKOK251[251];
__m128i ixOKOK257[257];
__m128i ixOKOK263[263];
__m128i ixOKOK269[269];
__m128i ixOKOK271[271];
__m128i ixOKOK277[277];


// selects elements from two vectors based on a selection mask
#define vec_sel(_X, _Y, _Z) _mm_blendv_epi8(_X, _Y, _Z)

// true if any element is not zero
#define continue_sito(_X) !_mm_testz_si128(_X,_X)

#define MAKE_OK(_X) \
  for(j=0;j<_X;j++) \
    OK##_X[j]=1; \
  for(j=(_X-23);j<=_X;j++) \
    OK##_X[(j*(STEP%_X))%_X]=0;


#define MAKE_OKOKx(_X) \
  for(j=0;j<_X;j++){ \
    aOKOK=0; \
    bOKOK=0; \
    for(jj=0;jj<64;jj++){ \
      if(SHIFT < maxshift) \
        aOKOK|=(((uint64_t)OK##_X[(j+(jj+SHIFT)*MOD)%_X])<<jj); \
      if(SHIFT+64 < maxshift) \
        bOKOK|=(((uint64_t)OK##_X[(j+(jj+SHIFT+64)*MOD)%_X])<<jj); \
    } \
    ixOKOK##_X[j] = _mm_set_epi64x( aOKOK, bOKOK ); \
  }


void *thr_func_sse41(void *arg) {

	thread_data_t *data = (thread_data_t *)arg;
	int err;
	int i43, i47, i53, i59;
	uint64_t n, n43, n47, n53, n59;

	err = pthread_mutex_lock(&lock1);
	if (err){
		fprintf(stderr, "ERROR: pthread_mutex_lock, code: %d\n", err);
		exit(EXIT_FAILURE);
	}
	uint64_t start = current_n43;
	uint64_t stop = start + thread_range;
	if(stop > numn43s){
		stop = numn43s;
	}
	current_n43 = stop;
	err = pthread_mutex_unlock(&lock1);
	if (err){
		fprintf(stderr, "ERROR: pthread_mutex_unlock, code: %d\n", err);
		exit(EXIT_FAILURE);
	}

	while(start < numn43s){
//		printf("thread: %d, start: %" PRId64 ", stop: %" PRId64 "\n",data->id, start, stop);
		for(;start<stop;++start){
			n43=n43_h[start];
			for(i43=(PRIME5-24);i43>0;i43--){
				n47=n43;
				for(i47=(PRIME6-24);i47>0;i47--){
					n53=n47;
					for(i53=(PRIME7-24);i53>0;i53--){
						n59=n53;
						__m128i r_numvec1 = _mm_set_epi16(REM(n59,61,6), REM(n59,67,7), REM(n59,71,7), REM(n59,73,7),
									 REM(n59,79,7), REM(n59,83,7), REM(n59,89,7), REM(n59,97,7));
						__m128i r_numvec2 = _mm_set_epi16(REM(n59,101,7), REM(n59,103,7), REM(n59,107,7), REM(n59,109,7),
									 REM(n59,113,7), REM(n59,127,7), REM(n59,131,8), REM(n59,137,8));

						for(i59=(PRIME8-24);i59>0;i59--){

							__m128i isito = _mm_and_si128( ixOKOK61[_mm_extract_epi16(r_numvec1, 7)], ixOKOK67[_mm_extract_epi16(r_numvec1, 6)] );
							isito = _mm_and_si128( isito, ixOKOK71[_mm_extract_epi16(r_numvec1, 5)] );
							isito = _mm_and_si128( isito, ixOKOK73[_mm_extract_epi16(r_numvec1, 4)] );
							isito = _mm_and_si128( isito, ixOKOK79[_mm_extract_epi16(r_numvec1, 3)] );
							isito = _mm_and_si128( isito, ixOKOK83[_mm_extract_epi16(r_numvec1, 2)] );
							isito = _mm_and_si128( isito, ixOKOK89[_mm_extract_epi16(r_numvec1, 1)] );
							isito = _mm_and_si128( isito, ixOKOK97[_mm_extract_epi16(r_numvec1, 0)] );
							isito = _mm_and_si128( isito, ixOKOK101[_mm_extract_epi16(r_numvec2, 7)] );
							isito = _mm_and_si128( isito, ixOKOK103[_mm_extract_epi16(r_numvec2, 6)] );
							isito = _mm_and_si128( isito, ixOKOK107[_mm_extract_epi16(r_numvec2, 5)] );
							isito = _mm_and_si128( isito, ixOKOK109[_mm_extract_epi16(r_numvec2, 4)] );
							isito = _mm_and_si128( isito, ixOKOK113[_mm_extract_epi16(r_numvec2, 3)] );
							isito = _mm_and_si128( isito, ixOKOK127[_mm_extract_epi16(r_numvec2, 2)] );
							isito = _mm_and_si128( isito, ixOKOK131[_mm_extract_epi16(r_numvec2, 1)] );
							isito = _mm_and_si128( isito, ixOKOK137[_mm_extract_epi16(r_numvec2, 0)] );
							if( continue_sito(isito) ){
								isito = _mm_and_si128( isito, ixOKOK139[REM(n59,139,8)] );
								isito = _mm_and_si128( isito, ixOKOK149[REM(n59,149,8)] );
								isito = _mm_and_si128( isito, ixOKOK151[REM(n59,151,8)] );
								isito = _mm_and_si128( isito, ixOKOK157[REM(n59,157,8)] );
								isito = _mm_and_si128( isito, ixOKOK163[REM(n59,163,8)] );
								isito = _mm_and_si128( isito, ixOKOK167[REM(n59,167,8)] );
								isito = _mm_and_si128( isito, ixOKOK173[REM(n59,173,8)] );
								isito = _mm_and_si128( isito, ixOKOK179[REM(n59,179,8)] );
								isito = _mm_and_si128( isito, ixOKOK181[REM(n59,181,8)] );
								isito = _mm_and_si128( isito, ixOKOK191[REM(n59,191,8)] );
								isito = _mm_and_si128( isito, ixOKOK193[REM(n59,193,8)] );
								isito = _mm_and_si128( isito, ixOKOK197[REM(n59,197,8)] );
								isito = _mm_and_si128( isito, ixOKOK199[REM(n59,199,8)] );
							if( continue_sito(isito) ){
								isito = _mm_and_si128( isito, ixOKOK211[REM(n59,211,8)] );
								isito = _mm_and_si128( isito, ixOKOK223[REM(n59,223,8)] );
								isito = _mm_and_si128( isito, ixOKOK227[REM(n59,227,8)] );
								isito = _mm_and_si128( isito, ixOKOK229[REM(n59,229,8)] );
								isito = _mm_and_si128( isito, ixOKOK233[REM(n59,233,8)] );
								isito = _mm_and_si128( isito, ixOKOK239[REM(n59,239,8)] );
								isito = _mm_and_si128( isito, ixOKOK241[REM(n59,241,8)] );
								isito = _mm_and_si128( isito, ixOKOK251[REM(n59,251,8)] );
								isito = _mm_and_si128( isito, ixOKOK257[REM(n59,257,9)] );
								isito = _mm_and_si128( isito, ixOKOK263[REM(n59,263,9)] );
								isito = _mm_and_si128( isito, ixOKOK269[REM(n59,269,9)] );
								isito = _mm_and_si128( isito, ixOKOK271[REM(n59,271,9)] );
								isito = _mm_and_si128( isito, ixOKOK277[REM(n59,277,9)] );
							if( continue_sito(isito) ){

								uint64_t sito[2];

								sito[0] = _mm_extract_epi64( isito, 1 );
								sito[1] = _mm_extract_epi64( isito, 0 );

								for(int ii=0;ii<2;++ii){
									if(sito[ii]){
										int b;
										uint64_t n;
										int bLimit, bStart;

										bLimit = 63 - __builtin_clzll(sito[ii]);
										bStart = __builtin_ctzll(sito[ii]);

										for (b = bStart; b <= bLimit; b++){
											if ((sito[ii] >> b) & 1)
											{
												n=n59+( b + data->SHIFT + (64*ii) )*MOD;

												if(n%7)
												if(n%11)
												if(n%13)
												if(n%17)
												if(n%19)
												if(n%23)
												if(OK281[n%281])
												if(OK283[n%283])
												if(OK293[n%293])
												if(OK307[n%307])
												if(OK311[n%311])
												if(OK313[n%313])
												if(OK317[n%317])
												if(OK331[n%331])
												if(OK337[n%337])
												if(OK347[n%347])
												if(OK349[n%349])
												if(OK353[n%353])
												if(OK359[n%359])
												if(OK367[n%367])
												if(OK373[n%373])
												if(OK379[n%379])
												if(OK383[n%383])
												if(OK389[n%389])
												if(OK397[n%397])
												if(OK401[n%401])
												if(OK409[n%409])
												if(OK419[n%419])
												if(OK421[n%421])
												if(OK431[n%431])
												if(OK433[n%433])
												if(OK439[n%439])
												if(OK443[n%443])
												if(OK449[n%449])
												if(OK457[n%457])
												if(OK461[n%461])
												if(OK463[n%463])
												if(OK467[n%467])
												if(OK479[n%479])
												if(OK487[n%487])
												if(OK491[n%491])
												if(OK499[n%499])
												if(OK503[n%503])
												if(OK509[n%509])
												if(OK521[n%521])
												if(OK523[n%523])
												if(OK541[n%541]){
													uint64_t m;
													int k;
													k=0; 
													m = n + data->STEP * 5;
													while(PrimeQ(m)){
														k++;
														m += data->STEP;
													}

													if(k>=10){
														m = n + data->STEP * 4;
														while(m>0&&PrimeQ(m)){
															k++;
															m -= data->STEP;
														}
													}

													if(k>=10){
														uint64_t first_term = m + data->STEP;

														err = pthread_mutex_lock(&lock2);
														if (err){
															fprintf(stderr, "ERROR: pthread_mutex_lock, code: %d\n", err);
															exit(EXIT_FAILURE);
														}
														ReportSolution(k, data->K, first_term);
														err = pthread_mutex_unlock(&lock2);
														if (err){
															fprintf(stderr, "ERROR: pthread_mutex_unlock, code: %d\n", err);
															exit(EXIT_FAILURE);
														}

													}
												}
											}
										}
									}
								}
							}}}

							n59 += data->S59;

							r_numvec1 = _mm_add_epi16(r_numvec1, svec1);
							r_numvec2 = _mm_add_epi16(r_numvec2, svec2);

							if(n59>=MOD){
								n59-=MOD;

								r_numvec1 = _mm_sub_epi16(r_numvec1, mvec1);
								__m128i addvec = _mm_add_epi16(r_numvec1, numvec1_1);
								r_numvec1 = vec_sel( r_numvec1, addvec, _mm_cmpgt_epi16( zerovec_avx, r_numvec1 ) );

								r_numvec2 = _mm_sub_epi16(r_numvec2, mvec2);
								addvec = _mm_add_epi16(r_numvec2, numvec1_2);
								r_numvec2 = vec_sel(r_numvec2, addvec, _mm_cmpgt_epi16( zerovec_avx, r_numvec2 ) );
							}
							__m128i subvec = _mm_sub_epi16(r_numvec1, numvec1_1);
							r_numvec1 = vec_sel(r_numvec1, subvec, _mm_cmpgt_epi16(r_numvec1, numvec2_1) );

							subvec = _mm_sub_epi16(r_numvec2, numvec1_2);
							r_numvec2 = vec_sel(r_numvec2, subvec, _mm_cmpgt_epi16(r_numvec2, numvec2_2) );
						   
						}     
						n53 += data->S53;
						if(n53>=MOD)n53-=MOD;
					}
					n47 += data->S47;
					if(n47>=MOD)n47-=MOD;
				}
				n43 += data->S43;
				if(n43>=MOD)n43-=MOD;
			}
		}


		err = pthread_mutex_lock(&lock1);
		if (err){
			fprintf(stderr, "ERROR: pthread_mutex_lock, code: %d\n", err);
			exit(EXIT_FAILURE);
		}
		start = current_n43;
		stop = start + thread_range;
		if(stop > numn43s){
			stop = numn43s;
		}
		current_n43 = stop;
		err = pthread_mutex_unlock(&lock1);
	}

	pthread_exit(NULL);

	return NULL;
}


void Search_sse41(int K, int startSHIFT, int K_COUNT, int K_DONE, int threads)
{ 
	int i3, i5, i31, i37, i41;
	int SHIFT;
	int maxshift = startSHIFT+640;
	uint64_t STEP;
	uint64_t n0;
	uint64_t S31, S37, S41, S43, S47, S53, S59;
	double d = (double)1.0 / (K_COUNT*numn43s*5);
	double dd;
	int j,jj,k;
	int err;
	uint64_t aOKOK,bOKOK;

	time_t start_time, finish_time, boinc_last, boinc_curr;

	if(boinc_standalone()){
		time (&start_time);
	}

	STEP=K*PRIM23;
	n0=(N0*(K%17835)+((N0*17835)%MOD)*(K/17835)+N30)%MOD;

	S31=(PRES2*(K%17835)+((PRES2*17835)%MOD)*(K/17835))%MOD;
	S37=(PRES3*(K%17835)+((PRES3*17835)%MOD)*(K/17835))%MOD;
	S41=(PRES4*(K%17835)+((PRES4*17835)%MOD)*(K/17835))%MOD;
	S43=(PRES5*(K%17835)+((PRES5*17835)%MOD)*(K/17835))%MOD;
	S47=(PRES6*(K%17835)+((PRES6*17835)%MOD)*(K/17835))%MOD;
	S53=(PRES7*(K%17835)+((PRES7*17835)%MOD)*(K/17835))%MOD;
	S59=(PRES8*(K%17835)+((PRES8*17835)%MOD)*(K/17835))%MOD;

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

	//quick loop vectors
	svec1		= _mm_set_epi16(S59%61, S59%67, S59%71, S59%73, S59%79, S59%83, S59%89, S59%97);
	svec2		= _mm_set_epi16(S59%101, S59%103, S59%107, S59%109, S59%113, S59%127, S59%131, S59%137);

	mvec1		= _mm_set_epi16(MOD%61, MOD%67, MOD%71, MOD%73, MOD%79, MOD%83, MOD%89, MOD%97);
	mvec2		= _mm_set_epi16(MOD%101, MOD%103, MOD%107, MOD%109, MOD%113, MOD%127, MOD%131, MOD%137);

	numvec1_1	= _mm_set_epi16(61, 67, 71, 73, 79, 83, 89, 97);
	numvec2_1	= _mm_set_epi16(60, 66, 70, 72, 78, 82, 88, 96);

	numvec1_2	= _mm_set_epi16(101, 103, 107, 109, 113, 127, 131, 137);
	numvec2_2	= _mm_set_epi16(100, 102, 106, 108, 112, 126, 130, 136);

	zerovec_avx	= _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0);

	// init OK arrays    
	MAKE_OK(61);
	MAKE_OK(67);
	MAKE_OK(71);
	MAKE_OK(73);
	MAKE_OK(79);
	MAKE_OK(83);
	MAKE_OK(89);
	MAKE_OK(97);
	MAKE_OK(101);
	MAKE_OK(103);
	MAKE_OK(107);
	MAKE_OK(109);
	MAKE_OK(113);
	MAKE_OK(127);
	MAKE_OK(131);
	MAKE_OK(137);
	MAKE_OK(139);
	MAKE_OK(149);
	MAKE_OK(151);
	MAKE_OK(157);
	MAKE_OK(163);
	MAKE_OK(167);
	MAKE_OK(173);
	MAKE_OK(179);
	MAKE_OK(181);
	MAKE_OK(191);
	MAKE_OK(193);
	MAKE_OK(197);
	MAKE_OK(199);
	MAKE_OK(211);
	MAKE_OK(223);
	MAKE_OK(227);
	MAKE_OK(229);
	MAKE_OK(233);
	MAKE_OK(239);
	MAKE_OK(241);
	MAKE_OK(251);
	MAKE_OK(257);
	MAKE_OK(263);
	MAKE_OK(269);
	MAKE_OK(271);
	MAKE_OK(277);
	MAKE_OK(281);
	MAKE_OK(283);
	MAKE_OK(293);
	MAKE_OK(307);
	MAKE_OK(311);
	MAKE_OK(313);
	MAKE_OK(317);
	MAKE_OK(331);
	MAKE_OK(337);
	MAKE_OK(347);
	MAKE_OK(349);
	MAKE_OK(353);
	MAKE_OK(359);
	MAKE_OK(367);
	MAKE_OK(373);
	MAKE_OK(379);
	MAKE_OK(383);
	MAKE_OK(389);
	MAKE_OK(397);
	MAKE_OK(401);
	MAKE_OK(409);
	MAKE_OK(419);
	MAKE_OK(421);
	MAKE_OK(431);
	MAKE_OK(433);
	MAKE_OK(439);
	MAKE_OK(443);
	MAKE_OK(449);
	MAKE_OK(457);
	MAKE_OK(461);
	MAKE_OK(463);
	MAKE_OK(467);
	MAKE_OK(479);
	MAKE_OK(487);
	MAKE_OK(491);
	MAKE_OK(499);
	MAKE_OK(503);
	MAKE_OK(509);
	MAKE_OK(521);
	MAKE_OK(523);
	MAKE_OK(541);
		
	int iteration = 0;

	// 10 shift
	for(SHIFT=startSHIFT; SHIFT<maxshift; SHIFT+=128){

		MAKE_OKOKx(61);
		MAKE_OKOKx(67);
		MAKE_OKOKx(71);
		MAKE_OKOKx(73);
		MAKE_OKOKx(79);
		MAKE_OKOKx(83);
		MAKE_OKOKx(89);
		MAKE_OKOKx(97);
		MAKE_OKOKx(101);
		MAKE_OKOKx(103);
		MAKE_OKOKx(107);
		MAKE_OKOKx(109);
		MAKE_OKOKx(113);
		MAKE_OKOKx(127);
		MAKE_OKOKx(131);
		MAKE_OKOKx(137);
		MAKE_OKOKx(139);
		MAKE_OKOKx(149);
		MAKE_OKOKx(151);
		MAKE_OKOKx(157);
		MAKE_OKOKx(163);
		MAKE_OKOKx(167);
		MAKE_OKOKx(173);
		MAKE_OKOKx(179);
		MAKE_OKOKx(181);
		MAKE_OKOKx(191);
		MAKE_OKOKx(193);
		MAKE_OKOKx(197);
		MAKE_OKOKx(199);
		MAKE_OKOKx(211);
		MAKE_OKOKx(223);
		MAKE_OKOKx(227);
		MAKE_OKOKx(229);
		MAKE_OKOKx(233);
		MAKE_OKOKx(239);
		MAKE_OKOKx(241);
		MAKE_OKOKx(251);
		MAKE_OKOKx(257);
		MAKE_OKOKx(263);
		MAKE_OKOKx(269);
		MAKE_OKOKx(271);
		MAKE_OKOKx(277);

		time(&boinc_last);

		pthread_t thr[threads];

		// create a thread_data_t argument array
		thread_data_t thr_data[threads];

		// initialize shared data
		current_n43 = 0;

		// create threads
		for (k = 0; k < threads; ++k) {
			thr_data[k].id = k;
			thr_data[k].K = K;
			thr_data[k].SHIFT = SHIFT;
			thr_data[k].STEP = STEP;
			thr_data[k].S43 = S43;
			thr_data[k].S47 = S47;
			thr_data[k].S53 = S53;
			thr_data[k].S59 = S59;
			err = pthread_create(&thr[k], NULL, thr_func_sse41, &thr_data[k]);
			if (err){
				fprintf(stderr, "ERROR: pthread_create, code: %d\n", err);
				exit(EXIT_FAILURE);
			}
		}

		// update BOINC fraction done every 2 sec and sleep the main thread
		err = pthread_mutex_lock(&lock1);
		if (err){
			fprintf(stderr, "ERROR: pthread_mutex_lock, code: %d\n", err);
			exit(EXIT_FAILURE);
		}
		uint32_t now = current_n43;
		err = pthread_mutex_unlock(&lock1);
		if (err){
			fprintf(stderr, "ERROR: pthread_mutex_unlock, code: %d\n", err);
			exit(EXIT_FAILURE);
		}
		while(now < numn43s){
			struct timespec sleep_time;
			sleep_time.tv_sec = 1;
			sleep_time.tv_nsec = 0;
			nanosleep(&sleep_time,NULL);

			time (&boinc_curr);
			if( ((int)boinc_curr - (int)boinc_last) > 1 ){
				dd = (double)(K_DONE*numn43s*5 + now + iteration*numn43s) * d;
				Progress(dd);
				boinc_last = boinc_curr;
			}

			err = pthread_mutex_lock(&lock1);
			if (err){
				fprintf(stderr, "ERROR: pthread_mutex_lock, code: %d\n", err);
				exit(EXIT_FAILURE);
			}
			now = current_n43;
			err = pthread_mutex_unlock(&lock1);
			if (err){
				fprintf(stderr, "ERROR: pthread_mutex_unlock, code: %d\n", err);
				exit(EXIT_FAILURE);
			}
		}

		// block until all threads complete
		for (k = 0; k < threads; ++k) {
			err = pthread_join(thr[k], NULL);
			if (err){
				fprintf(stderr, "ERROR: pthread_join, code: %d\n", err);
				exit(EXIT_FAILURE);
			}
		}

		++iteration;
		
	}

	if(boinc_standalone()){
		time(&finish_time);
		printf("Computation of K: %d complete in %d seconds\n", K, (int)finish_time - (int)start_time);
	}


}
