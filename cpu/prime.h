/* 
   prime.h -- Bryan Little 2-25-2020
   using GCC 64bit __int128


   PrimeQ_gen.h -- Geoffrey Reynolds, August 2009.

   This is an adaptation of Mark Rodenkirch's PPC64 mulmod code using
   only plain integer arithmetic available in C99.

*/

/* a*b mod p
   Assumes a,b < p < 2^63
*/
static uint64_t mulmod(uint64_t a, uint64_t b, uint64_t p, uint64_t magic, uint64_t shift)
{
	uint64_t ab0, ab1, mab00, mab01, mab10, mab11;
	uint64_t r, s0, s1, t;
	unsigned __int128 res;

	res = (unsigned __int128)a * b;
	ab0 = (uint64_t)res;
	ab1 = res >> 64;

	res = (unsigned __int128)magic * ab0;
	mab00 = (uint64_t)res;
	mab01 = res >> 64;

	res = (unsigned __int128)magic * ab1;
	mab10 = (uint64_t)res;
	mab11 = res >> 64;


	s0 = mab01 + mab10;
	s1 = mab11 + (s0 < mab01);

	t = (s0 >> shift) | (s1 << (64-shift));

	r = ab0 - t*p;

	if ((int64_t)r < 0)
		r += p;

	return r;
}


/* Returns 0 only if N is composite.
   Otherwise N is a strong probable prime to base a.
 */
static int strong_prp(uint64_t a, uint64_t N)
{
	uint64_t r, d, magic, shift;
	uint32_t s, t;

	/* getMagic */
	uint64_t two63 = 0x8000000000000000;
	uint64_t p = 63;
	uint64_t q2 = two63/N;
	uint64_t r2 = two63 - (q2 * N);
	uint64_t anc = two63 - 1 - r2;
	uint64_t q1 = two63/anc;
	uint64_t r1 = two63 - (q1 * anc);
	uint64_t delta;

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


	/* If N is prime and N = d*2^t+1, where d is odd, then either
		1.  a^d = 1 (mod N), or
		2.  a^(d*2^s) = -1 (mod N) for some s in 0 <= s < t    */

#ifdef __GNUC__
	t = __builtin_ctzll(N-1);
	d = N >> t;
#else
	for (d = N >> 1, t = 1; !(d & 1); d >>= 1, ++t);
#endif

	/* r <-- a^d mod N, assuming d odd */
	for (r = a, d >>= 1; d > 0; d >>= 1){
		a = mulmod(a,a,N,magic,shift);
		if (d & 1)
			r = mulmod(r,a,N,magic,shift);
	}

	/* Clause 1. and s = 0 case for clause 2. */
	if (r == 1 || r == N-1)
		return 1;

	/* 0 < s < t cases for clause 2. */
	for (s = 1; s < t; ++s)
		if ((r = mulmod(r,r,N,magic,shift)) == N-1)
			return 1;

	return 0;
}


/* Returns 0 only if N is composite.
   Otherwise N is a strong probable prime to base 2.
   For AP26, N can be assumed to have no prime divisors <= 541?
 */
int PrimeQ(int64_t N)
{
	return strong_prp(2,N);
}
