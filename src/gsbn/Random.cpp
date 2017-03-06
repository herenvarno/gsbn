#include "gsbn/Random.hpp"

namespace gsbn{

Random::Random(){
	random_device r;
	_rng_cpu = new default_random_engine(r());
	#ifndef CPU_ONLY
	_rng_gpu = new curandGenerator_t();
	//CURAND_CHECK(curandCreateGenerator(_rng_gpu, CURAND_RNG_QUASI_DEFAULT));
	CURAND_CHECK(curandCreateGenerator(_rng_gpu, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(*_rng_gpu, r()));
	#endif
}

Random::~Random(){
	if(_rng_cpu){
		delete _rng_cpu;
	}
	#ifndef CPU_ONLY
	if(_rng_gpu){
		delete _rng_gpu;
	}
	#endif
}

void Random::gen_uniform01_cpu(float *ptr, size_t size){
	uniform_real_distribution<float> dist(0, 1);
	for(size_t i=0; i<size; i++){
		*ptr = dist(*_rng_cpu);
		ptr++;
	}
}

void Random::gen_normal_cpu(float *ptr, size_t size, float mean, float sigma){
	normal_distribution<float> dist(mean, sigma);
	for(size_t i=0; i<size; i++){
		*ptr = dist(*_rng_cpu);
		ptr++;
	}
}

void Random::gen_poisson_cpu(unsigned int *ptr, size_t size, float mean){
	poisson_distribution<unsigned int> dist(mean);
	for(size_t i=0; i<size; i++){
		*ptr = dist(*_rng_cpu);
		ptr++;
	}
}

#ifndef CPU_ONLY

void Random::gen_uniform01_gpu(float *ptr, size_t size){
	CURAND_CHECK(curandGenerateUniform(*_rng_gpu, ptr, size));
}

void Random::gen_normal_gpu(float *ptr, size_t size, float mean, float sigma){
	CURAND_CHECK(curandGenerateNormal(*_rng_gpu, ptr, size, mean, sigma));
}

void Random::gen_poisson_gpu(unsigned int *ptr, size_t size, float mean){
	CURAND_CHECK(curandGeneratePoisson(*_rng_gpu, ptr, size, mean));
}

#endif


}
