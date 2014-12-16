#include "mex.h"
#include "gpu/mxGPUArray.h"

#include <cstdio>

__device__ int calcsum(int o, dim3 i, dim3 d)
{
	return (((o * d.z) + i.z) * d.y + i.y) * d.x + i.x;
}

__global__ void im2col(const float *const A, float *const B, int const m, int const n)
{
	int offset = calcsum(calcsum(0, blockIdx, gridDim), threadIdx, blockDim);
	int pffset = ((blockIdx.z * blockDim.z + threadIdx.z) * n + blockIdx.y + threadIdx.y) * m + blockIdx.x + threadIdx.x;

	B[offset] = A[pffset];
}

// TODO : Modify Error Messages, Add Stride Support
void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, mxArray const *prhs[])
{
	mxGPUArray const *A;
	mxGPUArray *B;
	float const *d_A;
	float *d_B;
	char const * const errId = "parallel:gpu:im2col_gpu:InvalidInput";
	char const * const errMsg = "Invalid input to MEX file.";
	mwSize nda;

	mxInitGPU();

	if ((nrhs < 2) || !(mxIsGPUArray(prhs[0]))) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}

	A = mxGPUCreateFromMxArray(prhs[0]);
	nda = mxGPUGetNumberOfDimensions(A);

	if (nda < 2 || nda > 4) {
		mexErrMsgIdAndTxt(errId, errMsg);
	} 
	
	if (mxGPUGetClassID(A) != mxSINGLE_CLASS) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}

	if (mxGetNumberOfElements(prhs[1]) < 1) {
		mexErrMsgIdAndTxt(errId, errMsg);
	}

	int m, n;

	if (mxGetNumberOfElements(prhs[1]) == 1) {
		m = n = mxGetScalar(prhs[1]);
	} else {
		auto *p = mxGetPr(prhs[1]);

		m = p[0];
		n = p[1];
	}

	auto *asz = mxGPUGetDimensions(A);

	mwSize mm = asz[0];
	mwSize nn = asz[1];
	mwSize c;
	mwSize N;

	if (nda < 3) {
		c = 1;
	} else {
		c = asz[2];
	}

	if (nda < 4) {
		N = 1;
	} else {
		N = asz[3];
	}

	mwSize s1 = mm - m + 1;
	mwSize s2 = nn - n + 1;

	mwSize bsz[6] = {m, n, c, s1, s2, N};

	d_A = (float const *)(mxGPUGetDataReadOnly(A));

	B = mxGPUCreateGPUArray(6, bsz,
		mxGPUGetClassID(A),
		mxGPUGetComplexity(A),
		MX_GPU_DO_NOT_INITIALIZE);
	d_B = (float *)(mxGPUGetData(B));

	im2col<<<dim3(bsz[3], bsz[4], bsz[5]), dim3(bsz[0], bsz[1], bsz[2])>>>(d_A, d_B, mm, nn);

	plhs[0] = mxGPUCreateMxArrayOnGPU(B);

	mxGPUDestroyGPUArray(A);
	mxGPUDestroyGPUArray(B);
}

