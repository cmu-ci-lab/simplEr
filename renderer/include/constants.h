/*
 * constants.h
 *
 *  Created on: Nov 24, 2015
 *      Author: igkiou
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#include <limits>
#include <stdint.h>

#if !defined(L1_CACHE_LINE_SIZE)
#define L1_CACHE_LINE_SIZE 64
#endif

typedef long int64;
#ifdef USE_DOUBLE_PRECISION
typedef double Float;
#define FPCONST(X) X
#else
typedef float Float;
#define FPCONST(X) X ## f
#endif

/* Choice of precision */
#ifdef USE_DOUBLE_PRECISION
//#define M_EPSILON	2.2204460492503131e-16
//#define M_MAX	1.7976931348623157e+308
//#define M_MIN	-1.7976931348623157e+308
#define M_EPSILON	1.19209290e-07
#define M_MAX	3.40282347e+38
#define M_MIN	-3.40282347e+38
#else
#define M_EPSILON	1.19209290e-07f
//const float M_EPSILON = std::numeric_limits<float>.epsilon();
#define M_MAX	3.40282347e+38f
#define M_MIN	-3.40282347e+38f
#endif

#ifdef M_PI
#undef M_PI
#endif

#ifdef USE_DOUBLE_PRECISION
#define M_PI         3.14159265358979323846
#define INV_PI       0.31830988618379067154
#define INV_TWOPI    0.15915494309189533577
#define INV_FOURPI   0.07957747154594766788
#define SQRT_TWO     1.41421356237309504880
#define INV_SQRT_TWO 0.70710678118654752440
#define HALF         0.5
#define ONE_SIXTH    0.16666666666666666667
#else
#define M_PI         3.14159265358979323846f
#define INV_PI       0.31830988618379067154f
#define INV_TWOPI    0.15915494309189533577f
#define INV_FOURPI   0.07957747154594766788f
#define SQRT_TWO     1.41421356237309504880f
#define INV_SQRT_TWO 0.70710678118654752440f
#define HALF         0.5f
#define ONE_SIXTH    0.16666666666666666667f
#endif

#endif /* CONSTANTS_H_ */
