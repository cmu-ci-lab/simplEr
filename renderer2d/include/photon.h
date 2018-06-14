/*
 * photon.h
 *
 *  Created on: Nov 21, 2015
 *      Author: igkiou
 */

#ifndef PHOTON_H_
#define PHOTON_H_

#include <stdio.h>
#include <vector>

#include "constants.h"
#include "image.h"
#include "phase.h"
#include "sampler.h"
#include "tvector.h"
#include "medium.h"
#include "scene.h"

#include <omp.h>

namespace photon {

class Renderer {
public:
	Renderer(const int maxDepth, const Float maxPathlength, const bool useDirect) :
			m_maxDepth(maxDepth),
			m_maxPathlength(maxPathlength),
			m_useDirect(useDirect) {
#ifndef NDEBUG
		std::cout << "maxDepth " << m_maxDepth << std::endl;
		std::cout << "maxPathlength " << m_maxPathlength << std::endl;
		std::cout << "useDirect " << m_useDirect << std::endl;
#endif
	}

	bool scatterOnce(tvec::Vec3f &p, tvec::Vec3f &d, Float &dist, Float &cosTheta,
					const scn::Scene &scene, const med::Medium &medium,
					smp::Sampler &sampler) const;

	void scatter(const tvec::Vec3f &p, const tvec::Vec3f &d,
				const scn::Scene &scene, const med::Medium &medium,
				smp::Sampler &sampler, image::SmallImage &img, Float weight) const;

	void scatterDeriv(const tvec::Vec3f &p, const tvec::Vec3f &d,
					const scn::Scene &scene, const med::Medium &medium,
					smp::Sampler &sampler, image::SmallImage &img,
					image::SmallImage &dSigmaT, image::SmallImage &dAlbedo,
					image::SmallImage &dGVal, Float weight) const;

	bool scatterOnceWeight(tvec::Vec3f &p, tvec::Vec3f &d, Float &weight,
							Float &dist, Float &cosTheta, const scn::Scene &scene,
							const med::Medium &medium, const med::Medium &samplingMedium,
							smp::Sampler &sampler) const;

	void scatterDerivWeight(const tvec::Vec3f &p, const tvec::Vec3f &d,
						const scn::Scene &scene, const med::Medium &medium,
						const med::Medium &samplingMedium,
						smp::Sampler &sampler, image::SmallImage &img,
						image::SmallImage &dSigmaT, image::SmallImage &dAlbedo,
						image::SmallImage &dGVal, Float weight) const;


//	bool scatterOnceWeight(tvec::Vec3f &p, tvec::Vec3f &d, Float &weight,
//					const scn::Scene &scene, const med::Medium &medium, smp::Sampler &sampler) const;
//
//	void scatterWeight(const tvec::Vec3f &p, const tvec::Vec3f &d,
//					const scn::Scene &scene, const med::Medium &medium,
//					smp::Sampler &sampler, image::SmallImage &img, Float weight) const;

	inline Float getMoveStep(const med::Medium &medium, smp::Sampler &sampler) const {
		return -medium.getMfp() * std::log(sampler());
	}

	inline Float getWeight(const med::Medium &, const scn::Scene &scene, const int64 numPhotons) const {
		return scene.getAreaSource().getLi() * scene.getFresnelTrans() / static_cast<Float>(numPhotons);
	}

	void renderImage(image::SmallImage &img0,
					const med::Medium &medium, const scn::Scene &scene,
					const int64 numPhotons) const;

	void renderDerivImage(image::SmallImage &img0, image::SmallImage &dSigmaT0,
					image::SmallImage &dAlbedo0, image::SmallImage &dGVal0,
					const med::Medium &medium, const scn::Scene &scene,
					const int64 numPhotons) const;

	void renderDerivImageWeight(image::SmallImage &img0, image::SmallImage &dSigmaT0,
					image::SmallImage &dAlbedo0, image::SmallImage &dGVal0,
					const med::Medium &medium, const med::Medium &samplingMedium,
					const scn::Scene &scene, const int64 numPhotons) const;

	inline int getMaxDepth() const {
		return m_maxDepth;
	}

	inline Float getMaxPathlength() const {
		return m_maxPathlength;
	}

	inline bool getUseDirect() const {
		return m_useDirect;
	}

	~Renderer() { }

protected:
	int m_maxDepth;
	Float m_maxPathlength;
	bool m_useDirect;
};

}	/* namespace photon */

#endif /* PHOTON_H_ */
