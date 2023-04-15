/* 
	checkn.cl
	tests primality of each term of the AP sequence
	test is good to 2^64-1

*/



// r0 + 2^64 * r1 = a * b
inline ulong2 mul_wide(const ulong a, const ulong b)
{
	ulong2 r;

#ifdef __NV_CL_C_VERSION
	const uint a0 = (uint)(a), a1 = (uint)(a >> 32);
	const uint b0 = (uint)(b), b1 = (uint)(b >> 32);

	uint c0 = a0 * b0, c1 = mul_hi(a0, b0), c2, c3;

	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c1) : "r" (a0), "r" (b1), "r" (c1));
	asm volatile ("madc.hi.u32 %0, %1, %2, 0;" : "=r" (c2) : "r" (a0), "r" (b1));

	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c2) : "r" (a1), "r" (b1), "r" (c2));
	asm volatile ("madc.hi.u32 %0, %1, %2, 0;" : "=r" (c3) : "r" (a1), "r" (b1));

	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c1) : "r" (a1), "r" (b0), "r" (c1));
	asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r" (c2) : "r" (a1), "r" (b0), "r" (c2));
	asm volatile ("addc.u32 %0, %1, 0;" : "=r" (c3) : "r" (c3));

	r.s0 = upsample(c1, c0); r.s1 = upsample(c3, c2);
#else
	r.s0 = a * b; r.s1 = mul_hi(a, b);
#endif

	return r;
}


inline ulong invert(ulong p)
{
	ulong p_inv = 1, prev = 0;
	while (p_inv != prev) { prev = p_inv; p_inv *= 2 - p * p_inv; }
	return p_inv;
}


inline ulong montMul(ulong a, ulong b, ulong p, ulong q)
{
	ulong2 ab = mul_wide(a,b);

	ulong m = ab.s0 * q;

	ulong mp = mul_hi(m,p);

	ulong r = ab.s1 - mp;

	return ( ab.s1 < mp ) ? r + p : r;
}


inline ulong add(ulong a, ulong b, ulong p)
{
	ulong r;

	ulong c = (a >= p - b) ? p : 0;

	r = a + b - c;

	return r;
}

// strong probable prime to base 2
inline bool strong_prp(ulong N)
{
	ulong nmo = N-1;
	int t = 63 - clz(nmo & -nmo);	// this is ctz
	ulong exp = N >> t;
	ulong curBit = 0x8000000000000000;
	curBit >>= ( clz(exp) + 1 );

	ulong q = invert(N);
	ulong one = (-N) % N;
	ulong a = add(one, one, N); 	// two, in montgomery form
	nmo = N - one;  		// N-1 in montgomery form

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



/*
	main prime sequence checking kernel
*/
__kernel void checkn(__global ulong * n_result, ulong STEP, __global int * sol_k, __global ulong * sol_val, __global int * counter){

	int gid = get_global_id(0);

	if(gid < counter[0]){

		ulong n = n_result[gid];

		ulong m = n + STEP*5;

		if(m < n){  // software limit
			atomic_or(&counter[3], 1);
		}

		int k=0;

		// forward
		while(strong_prp( m )){
			m += STEP;
			++k;
			if(m < n){  // software limit
				atomic_or(&counter[3], 1);
				break;
			}
		}

		if(k >= 10){
			m = n + STEP*4;
			ulong start = m;

			// reverse
			while(strong_prp( m )){
				m -= STEP;
				++k;
				if(m > start)break;  // m < 0
			}

			// AP length >= 10 store to results
			int index = atomic_inc(&counter[2]);
			sol_k[index] = k;
			sol_val[index] = m+STEP;
		}

	}

	// store largest ncount
	if(gid == 0){
		int nc = counter[0];
		if(nc > counter[1]){
			counter[1] = nc;
		}
	}
}
