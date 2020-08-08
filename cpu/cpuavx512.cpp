/* cpuavx512.cpp --

   AVX512 support March 23 2020 by Bryan Little

   See http://www.math.uni.wroc.pl/~jwr/AP26/AP26v3.pdf for information
   about how the algorithm works and for the copyleft notice.
*/

#include <x86intrin.h>
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

#include "cpuconst.h"

__m256i svec, mvec, numvec1, numvec2, zerovec; // avx2, avx512

// char arrays total 23693 bytes
char OK61[61];
char OK67[67];
char OK71[71];
char OK73[73];
char OK79[79];
char OK83[83];
char OK89[89];
char OK97[97];
char OK101[101];
char OK103[103];
char OK107[107];
char OK109[109];
char OK113[113];
char OK127[127];
char OK131[131];
char OK137[137];
char OK139[139];
char OK149[149];
char OK151[151];
char OK157[157];
char OK163[163];
char OK167[167];
char OK173[173];
char OK179[179];
char OK181[181];
char OK191[191];
char OK193[193];
char OK197[197];
char OK199[199];
char OK211[211];
char OK223[223];
char OK227[227];
char OK229[229];
char OK233[233];
char OK239[239];
char OK241[241];
char OK251[251];
char OK257[257];
char OK263[263];
char OK269[269];
char OK271[271];
char OK277[277];
char OK281[281];
char OK283[283];
char OK293[293];
char OK307[307];
char OK311[311];
char OK313[313];
char OK317[317];
char OK331[331];
char OK337[337];
char OK347[347];
char OK349[349];
char OK353[353];
char OK359[359];
char OK367[367];
char OK373[373];
char OK379[379];
char OK383[383];
char OK389[389];
char OK397[397];
char OK401[401];
char OK409[409];
char OK419[419];
char OK421[421];
char OK431[431];
char OK433[433];
char OK439[439];
char OK443[443];
char OK449[449];
char OK457[457];
char OK461[461];
char OK463[463];
char OK467[467];
char OK479[479];
char OK487[487];
char OK491[491];
char OK499[499];
char OK503[503];
char OK509[509];
char OK521[521];
char OK523[523];
char OK541[541];

__m512i xxOKOK61[61];
__m512i xxOKOK67[67];
__m512i xxOKOK71[71];
__m512i xxOKOK73[73];
__m512i xxOKOK79[79];
__m512i xxOKOK83[83];
__m512i xxOKOK89[89];
__m512i xxOKOK97[97];
__m512i xxOKOK101[101];
__m512i xxOKOK103[103];
__m512i xxOKOK107[107];
__m512i xxOKOK109[109];
__m512i xxOKOK113[113];
__m512i xxOKOK127[127];
__m512i xxOKOK131[131];
__m512i xxOKOK137[137];
__m512i xxOKOK139[139];
__m512i xxOKOK149[149];
__m512i xxOKOK151[151];
__m512i xxOKOK157[157];
__m512i xxOKOK163[163];
__m512i xxOKOK167[167];
__m512i xxOKOK173[173];
__m512i xxOKOK179[179];
__m512i xxOKOK181[181];
__m512i xxOKOK191[191];
__m512i xxOKOK193[193];
__m512i xxOKOK197[197];
__m512i xxOKOK199[199];
__m512i xxOKOK211[211];
__m512i xxOKOK223[223];
__m512i xxOKOK227[227];
__m512i xxOKOK229[229];
__m512i xxOKOK233[233];
__m512i xxOKOK239[239];
__m512i xxOKOK241[241];
__m512i xxOKOK251[251];
__m512i xxOKOK257[257];
__m512i xxOKOK263[263];
__m512i xxOKOK269[269];
__m512i xxOKOK271[271];
__m512i xxOKOK277[277];


// selects elements from two vectors based on a selection mask
#define vec_sel(_X, _Y, _Z) _mm256_blendv_epi8(_X, _Y, _Z)


// true if any element is not zero
#define continue_sito(_X) _mm512_cmpneq_epi64_mask(_X, _mm512_setzero_si512())


#define MAKE_OK(_X) \
  for(j=0;j<_X;j++) \
    OK##_X[j]=1; \
  for(j=(_X-23);j<=_X;j++) \
    OK##_X[(j*(STEP%_X))%_X]=0;


#define MAKE_OKOKxx(_X) \
  for(j=0;j<_X;j++){ \
    aOKOK=0; \
    bOKOK=0; \
    cOKOK=0; \
    dOKOK=0; \
    eOKOK=0; \
    fOKOK=0; \
    gOKOK=0; \
    hOKOK=0; \
    for(jj=0;jj<64;jj++){ \
      if(SHIFT < maxshift) \
        aOKOK|=(((int64_t)OK##_X[(j+(jj+SHIFT)*MOD)%_X])<<jj); \
      if(SHIFT+64 < maxshift) \
        bOKOK|=(((int64_t)OK##_X[(j+(jj+SHIFT+64)*MOD)%_X])<<jj); \
      if(SHIFT+128 < maxshift) \
        cOKOK|=(((int64_t)OK##_X[(j+(jj+SHIFT+128)*MOD)%_X])<<jj); \
      if(SHIFT+192 < maxshift) \
        dOKOK|=(((int64_t)OK##_X[(j+(jj+SHIFT+192)*MOD)%_X])<<jj); \
      if(SHIFT+256 < maxshift) \
        eOKOK|=(((int64_t)OK##_X[(j+(jj+SHIFT+256)*MOD)%_X])<<jj); \
      if(SHIFT+320 < maxshift) \
        fOKOK|=(((int64_t)OK##_X[(j+(jj+SHIFT+320)*MOD)%_X])<<jj); \
      if(SHIFT+384 < maxshift) \
        gOKOK|=(((int64_t)OK##_X[(j+(jj+SHIFT+384)*MOD)%_X])<<jj); \
      if(SHIFT+448 < maxshift) \
        hOKOK|=(((int64_t)OK##_X[(j+(jj+SHIFT+448)*MOD)%_X])<<jj); \
    } \
    xxOKOK##_X[j] = _mm512_set_epi64( aOKOK, bOKOK, cOKOK, dOKOK, eOKOK, fOKOK, gOKOK, hOKOK ); \
  }


void *thr_func_avx512(void *arg) {

	thread_data_t *data = (thread_data_t *)arg;
	int err;
	int i43, i47, i53, i59;
	int64_t n, n43, n47, n53, n59;

	err = pthread_mutex_lock(&lock1);
	if (err){
		fprintf(stderr, "ERROR: pthread_mutex_lock, code: %d\n", err);
		exit(EXIT_FAILURE);
	}
	int64_t start = current_n43;
	int64_t stop = start + thread_range;
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
						__m256i rvec = _mm256_set_epi16(REM(n59,61,6), REM(n59,67,7), REM(n59,71,7), REM(n59,73,7), REM(n59,79,7), REM(n59,83,7), REM(n59,89,7), REM(n59,97,7),
								REM(n59,101,7), REM(n59,103,7), REM(n59,107,7), REM(n59,109,7), REM(n59,113,7), REM(n59,127,7), REM(n59,131,8), REM(n59,137,8));

						for(i59=(PRIME8-24);i59>0;i59--){

							__m512i dsito = _mm512_and_epi64( xxOKOK61[_mm256_extract_epi16(rvec, 15)], xxOKOK67[_mm256_extract_epi16(rvec, 14)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK71[_mm256_extract_epi16(rvec, 13)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK73[_mm256_extract_epi16(rvec, 12)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK79[_mm256_extract_epi16(rvec, 11)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK83[_mm256_extract_epi16(rvec, 10)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK89[_mm256_extract_epi16(rvec, 9)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK97[_mm256_extract_epi16(rvec, 8)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK101[_mm256_extract_epi16(rvec, 7)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK103[_mm256_extract_epi16(rvec, 6)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK107[_mm256_extract_epi16(rvec, 5)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK109[_mm256_extract_epi16(rvec, 4)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK113[_mm256_extract_epi16(rvec, 3)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK127[_mm256_extract_epi16(rvec, 2)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK131[_mm256_extract_epi16(rvec, 1)] );
							dsito = _mm512_and_epi64( dsito, xxOKOK137[_mm256_extract_epi16(rvec, 0)] );
							if( continue_sito(dsito) ){
								dsito = _mm512_and_epi64( dsito, xxOKOK139[REM(n59,139,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK149[REM(n59,149,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK151[REM(n59,151,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK157[REM(n59,157,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK163[REM(n59,163,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK167[REM(n59,167,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK173[REM(n59,173,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK179[REM(n59,179,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK181[REM(n59,181,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK191[REM(n59,191,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK193[REM(n59,193,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK197[REM(n59,197,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK199[REM(n59,199,8)] );
							if( continue_sito(dsito) ){
								dsito = _mm512_and_epi64( dsito, xxOKOK211[REM(n59,211,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK223[REM(n59,223,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK227[REM(n59,227,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK229[REM(n59,229,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK233[REM(n59,233,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK239[REM(n59,239,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK241[REM(n59,241,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK251[REM(n59,251,8)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK257[REM(n59,257,9)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK263[REM(n59,263,9)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK269[REM(n59,269,9)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK271[REM(n59,271,9)] );
								dsito = _mm512_and_epi64( dsito, xxOKOK277[REM(n59,277,9)] );
							if( continue_sito(dsito) ){

								__m256i four_sito = _mm512_extracti64x4_epi64( dsito, 1 );
								__m256i four_sito2 = _mm512_extracti64x4_epi64( dsito , 0 );

								int64_t sito[8];

								sito[0] = _mm256_extract_epi64( four_sito, 3 );
								sito[1] = _mm256_extract_epi64( four_sito, 2 );
								sito[2] = _mm256_extract_epi64( four_sito, 1 );
								sito[3] = _mm256_extract_epi64( four_sito, 0 );
								sito[4] = _mm256_extract_epi64( four_sito2, 3 );
								sito[5] = _mm256_extract_epi64( four_sito2, 2 );
								sito[6] = _mm256_extract_epi64( four_sito2, 1 );
								sito[7] = _mm256_extract_epi64( four_sito2, 0 );


								for(int ii=0;ii<8;++ii){
									if(sito[ii]){
										int b;
										int64_t n;
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
													int64_t m;
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
														int64_t first_term = m + data->STEP;

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

							rvec = _mm256_add_epi16(rvec, svec);

							if(n59>=MOD){
								n59-=MOD;

								rvec = _mm256_sub_epi16(rvec, mvec);
								__m256i addvec = _mm256_add_epi16(rvec, numvec1);
								rvec = vec_sel( rvec, addvec, _mm256_cmpgt_epi16( zerovec, rvec ) );

							}

							__m256i subvec = _mm256_sub_epi16(rvec, numvec1);
							rvec = vec_sel(rvec, subvec, _mm256_cmpgt_epi16(rvec, numvec2) );
						   
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


void Search_avx512(int K, int startSHIFT, int K_COUNT, int K_DONE, int threads)
{ 
	int i3, i5, i31, i37, i41;
	int SHIFT;
	int maxshift = startSHIFT+640;
	int64_t STEP;
	int64_t n0;
	int64_t S31, S37, S41, S43, S47, S53, S59;
	double d = (double)1.0 / (K_COUNT*numn43s*2);
	double dd;
	int j,jj,k;
	int err;
	int64_t aOKOK,bOKOK,cOKOK,dOKOK,eOKOK,fOKOK,gOKOK,hOKOK;

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
	svec = _mm256_set_epi16(S59%61, S59%67, S59%71, S59%73, S59%79, S59%83, S59%89, S59%97,
				S59%101, S59%103, S59%107, S59%109, S59%113, S59%127, S59%131, S59%137);

	mvec = _mm256_set_epi16(MOD%61, MOD%67, MOD%71, MOD%73, MOD%79, MOD%83, MOD%89, MOD%97,
				MOD%101, MOD%103, MOD%107, MOD%109, MOD%113, MOD%127, MOD%131, MOD%137);

	numvec1	= _mm256_set_epi16(61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137);

	numvec2	= _mm256_set_epi16(60, 66, 70, 72, 78, 82, 88, 96, 100, 102, 106, 108, 112, 126, 130, 136);
						
	zerovec	= _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

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
	for(SHIFT=startSHIFT; SHIFT<maxshift; SHIFT+=512){

		MAKE_OKOKxx(61);
		MAKE_OKOKxx(67);
		MAKE_OKOKxx(71);
		MAKE_OKOKxx(73);
		MAKE_OKOKxx(79);
		MAKE_OKOKxx(83);
		MAKE_OKOKxx(89);
		MAKE_OKOKxx(97);
		MAKE_OKOKxx(101);
		MAKE_OKOKxx(103);
		MAKE_OKOKxx(107);
		MAKE_OKOKxx(109);
		MAKE_OKOKxx(113);
		MAKE_OKOKxx(127);
		MAKE_OKOKxx(131);
		MAKE_OKOKxx(137);
		MAKE_OKOKxx(139);
		MAKE_OKOKxx(149);
		MAKE_OKOKxx(151);
		MAKE_OKOKxx(157);
		MAKE_OKOKxx(163);
		MAKE_OKOKxx(167);
		MAKE_OKOKxx(173);
		MAKE_OKOKxx(179);
		MAKE_OKOKxx(181);
		MAKE_OKOKxx(191);
		MAKE_OKOKxx(193);
		MAKE_OKOKxx(197);
		MAKE_OKOKxx(199);
		MAKE_OKOKxx(211);
		MAKE_OKOKxx(223);
		MAKE_OKOKxx(227);
		MAKE_OKOKxx(229);
		MAKE_OKOKxx(233);
		MAKE_OKOKxx(239);
		MAKE_OKOKxx(241);
		MAKE_OKOKxx(251);
		MAKE_OKOKxx(257);
		MAKE_OKOKxx(263);
		MAKE_OKOKxx(269);
		MAKE_OKOKxx(271);
		MAKE_OKOKxx(277);

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
			err = pthread_create(&thr[k], NULL, thr_func_avx512, &thr_data[k]);
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
				dd = (double)(K_DONE*numn43s*2 + now + iteration*numn43s) * d;
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
