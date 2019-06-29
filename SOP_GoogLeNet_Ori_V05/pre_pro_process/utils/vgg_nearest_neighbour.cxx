#include "mex.h"
#include <float.h>
#include <memory.h>

// [ind, d2] = vgg_nearest_neighbour(X, Y)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	int				nx, ny, dim;
	const double	*X, *Y, *py;
	int				*ind;
	double			*d2;
	double			d, dmin;
	int				i, j, k;

	if (nrhs != 2)
		mexErrMsgTxt("two input arguments expected.");
	if (nlhs != 2)
		mexErrMsgTxt("two output arguments expected.");

	if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) ||
		mxGetNumberOfDimensions(prhs[0]) != 2)
		mexErrMsgTxt("input 1 (X) must be a real double matrix");

	dim = mxGetM(prhs[0]);
	nx = mxGetN(prhs[0]);

	if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) ||
		mxGetNumberOfDimensions(prhs[1]) != 2 ||
		mxGetM(prhs[1]) != dim)
		mexErrMsgTxt("input 2 (Y) must be a real double matrix compatible with input 1 (X)");

	ny = mxGetN(prhs[1]);

	plhs[0] = mxCreateNumericMatrix(nx, 1, mxINT32_CLASS, mxREAL);
	ind = (int *) mxGetData(plhs[0]);

	plhs[1] = mxCreateDoubleMatrix(nx, 1, mxREAL);
	d2 = mxGetPr(plhs[1]);

	X = mxGetPr(prhs[0]);
	Y = mxGetPr(prhs[1]);

	for (i = 1; i <= nx; i++, X += dim, ind++)
	{
		dmin = DBL_MAX;
		*ind = 0;
		for (j = 1, py = Y; j <= ny; j++, py += dim)
		{
			d = 0.0;
			for (k = 0; k < dim; k++) {
			   double diff = (X[k] - py[k]);
			   d += diff*diff;
			   if (d > dmin)
			     break;
			}
			if (d < dmin)
			{
				dmin = d;
				*ind = j;
			}
		}

		*(d2++) = dmin;
	}
}
