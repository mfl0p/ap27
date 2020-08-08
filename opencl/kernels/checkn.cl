/* PrimeQ -- Geoffrey Reynolds, September 11 2008.

   See http://www.math.uni.wroc.pl/~jwr/AP26/AP26.zip for the sample
   implementation containing the rest of the source required.

   See http://www.math.uni.wroc.pl/~jwr/AP26/AP26v3.pdf for information
   about how the algorithm works and for the copyleft notice.

   PrimeQ(N) is limited to N < 2^63 - 1. This is adequate for the AP26 search.


	***************
	Bryan Little 
	April 2020 removed mulmod function
	May 2014 OpenCL conversion - checkn.cl


*/



/* Returns 0 only if N is composite.
   Otherwise N is a strong probable prime to base 2.
   For AP26, N can be assumed to have no prime divisors <= 541?
 */
int strong_prp(ulong N)
{
	ulong magic, shift, r, d;
	int s, t;
	ulong a = 2; // prp to base a
	ulong ab0, ab1, mab01, mab10, mab11;
	ulong res, s0, s1, tt;
	int retval = 0;

	/* getMagic */
	ulong two63 = 0x8000000000000000;
	ulong p = 63;
	ulong q2 = two63/N;
	ulong r2 = two63 - (q2 * N);
	ulong anc = two63 - 1 - r2;
	ulong q1 = two63/anc;
	ulong r1 = two63 - (q1 * anc);
	ulong delta;

	do {
		++p;
		q1 = 2*q1;
		r1 = 2*r1; 
		q2 = 2*q2;
		r2 = 2*r2;
		if (r1 >= anc) {
			++q1;
			r1-=anc;
      		}
		if (r2 >= N) {
			++q2;
			r2-=N;
      		}
		delta = N - r2;
   	} while (q1 < delta || (q1 == delta && r1 == 0));

	shift = p - 64;
	magic = q2 + 1;
	/* end getMagic */

	d = N >> 1;
	t = 1;

	while( !(d & 1) ){
		d >>= 1;
		++t;
	}

	r = a;
	d >>= 1;

  	/* r <-- a^d mod N, assuming d odd */
	while( d > 0 )
	{
		// a = mulmod(a,a,N,magic,shift);
		ab1 = mul_hi(a,a);
		ab0 = a*a;
		mab01 = mul_hi(magic,ab0);
		mab10 = magic * ab1;
		mab11 = mul_hi(magic,ab1);

		s0 = mab01 + mab10;
		s1 = mab11 + (s0 < mab01);
		tt = (s0 >> shift) | (s1 << (64-shift));

		res = ab0 - tt * N;

		if((long)res < 0)
			res += N;
	  
		a = res;
		// end mulmod

		if(d & 1){

			// r = mulmod(r,a,N,magic,shift);
			ab1 = mul_hi(r,a);
			ab0 = r*a;
			mab01 = mul_hi(magic,ab0);
			mab10 = magic * ab1;
			mab11 = mul_hi(magic,ab1);

			s0 = mab01 + mab10;
			s1 = mab11 + (s0 < mab01);
			tt = (s0 >> shift) | (s1 << (64-shift));

			res = ab0 - tt * N;

			if((long)res < 0)
				res += N;
		  
			r = res;
			// end mulmod
		}

		d >>= 1;
	}

	/* Clause 1. and s = 0 case for clause 2. */
	if (r == 1 || r == N-1){
		retval = 1;
	}

	/* 0 < s < t cases for clause 2. */
	for (s = 1; !retval && s < t; ++s){

		// r = mulmod(r,r,N,magic,shift);
		ab1 = mul_hi(r,r);
		ab0 = r*r;
		mab01 = mul_hi(magic,ab0);
		mab10 = magic * ab1;
		mab11 = mul_hi(magic,ab1);

		s0 = mab01 + mab10;
		s1 = mab11 + (s0 < mab01);
		tt = (s0 >> shift) | (s1 << (64-shift));

		res = ab0 - tt * N;

		if((long)res < 0)
			res += N;
	  
		r = res;
		// end mulmod

		if(r == N-1){
	    		retval = 1;
		}
	}

	return retval;
}


/*
	main prime sequence checking kernel
*/
__kernel void checkn(__global long *n_result, long STEP, __global int *sol_k, __global long *sol_val, __global int *counter){

	int i = get_global_id(0);

	if(i<counter[0]){

		long n = n_result[i];
		long m = n+STEP*5;
		int forward = 1;
		int backward = 1;
		int k=0;

		while(forward || backward){

			if(strong_prp( (ulong)m )){
				if(forward){
					m+=STEP;
				}
				else{
					m-=STEP;
					if(m<=0){
						backward=0;
					}
				}
				k++;
			}
			else{
				if(forward){
					forward=0;
					if(k>=10){
						m=n+STEP*4;
					}
					else{
						backward=0;
					}
				}
				else{
					backward=0;
				}
			}
		}

		if(k>=10){
			int index = atomic_add(&counter[2], 1);
			sol_k[index] = k;
			sol_val[index] = m+STEP;
		}

	}

	// store largest ncount
	if(i == 0){
		int nc = counter[0];
		if(nc > counter[1]){
			counter[1] = nc;
		}
	}
}
