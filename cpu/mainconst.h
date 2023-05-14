// mainconst.h

extern void Search_avx512(int K, int startSHIFT, int K_COUNT, int K_DONE, int threads);
extern void Search_avx2(int K, int startSHIFT, int K_COUNT, int K_DONE, int threads);
extern void Search_avx(int K, int startSHIFT, int K_COUNT, int K_DONE, int threads);
extern void Search_sse41(int K, int startSHIFT, int K_COUNT, int K_DONE, int threads);
extern void Search_sse2(int K, int startSHIFT, int K_COUNT, int K_DONE, int threads);

#define numn43s	10840

#define PRIM23	UINT64_C(223092870)
#define PRIME1	29
#define PRIME2	31
#define PRIME3	37
#define PRIME4	41
#define PRIME5	43
#define PRIME6	47
#define PRIME7	53
#define PRIME8	59
#define MOD		UINT64_C(258559632607830)
#define N0		UINT64_C(106990415896110)
#define N30		UINT64_C(94805198622871)
#define S3		UINT64_C(172373088405220)
#define S5		UINT64_C(51711926521566)
#define PRES2	UINT64_C(16681266619860)
#define PRES3	UINT64_C(181690552643340)
#define PRES4	UINT64_C(132432982555230)
#define PRES5	UINT64_C(126273308948010)
#define PRES6	UINT64_C(115526644356690)
#define PRES7	UINT64_C(48784836341100)
#define PRES8	UINT64_C(100794433050510)
















