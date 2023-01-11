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


