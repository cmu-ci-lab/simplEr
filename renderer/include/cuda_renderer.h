/*
 * cuda_renderer.h
 *
 *  Created on: Aug 2, 2021
 *      Author: Andre
 */


#ifndef CUDA_RENDERER_H_
#define CUDA_RENDERER_H_

#include <curand.h>

#include "image.h"
#include "sampler.h"

#include "cuda_scene.cuh"

namespace cuda {

typedef unsigned int CudaSeedType;

typedef struct threadScatterState {
	int shouldContinue; // 1 if should continue, 0 othws
	TVector3<Float> p;
	TVector3<Float> d;
	Float totalDist;
	Float dist;
	int depth;
	Float totalOpticalDistance;
	Float scaling;
} threadState_t;


class CudaRenderer {

public:

    CudaRenderer(const int maxDepth, const Float maxPathlength, const bool useDirect, const bool useAngularSampling) :
			maxDepth(maxDepth),
			maxPathlength(maxPathlength),
			useDirect(useDirect),
			useAngularSampling(useAngularSampling) {};

    ~CudaRenderer();

    void renderImage(image::SmallImage& target, const med::Medium &medium, const scn::Scene<tvec::TVector3> &scene, int numPhotons);

private:

    Float *cudaImage;
    Scene *cudaScene;
    Medium *cudaMedium;
    threadState_t* continueData;

    inline Float getWeight(const med::Medium &, const scn::Scene<tvec::TVector3> &scene,
                            const int64 numPhotons) {
        return scene.getAreaSource().getLi() * scene.getFresnelTrans()
                / static_cast<Float>(numPhotons);
    }
    void setup(image::SmallImage& target, const med::Medium &medium, const scn::Scene<tvec::TVector3> &scene, int numPhotons);
    void cleanup();

    /* Host memory*/
    Float *image;

    int maxDepth;
    Float maxPathlength;
    bool useDirect, useAngularSampling;
};

}


#endif /* CUDA_RENDERER_H_ */
