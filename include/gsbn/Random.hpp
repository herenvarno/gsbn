#ifndef __GSBN_RANDOM_HPP__
#define __GSBN_RANDOM_HPP__

#include <random>
#include "gsbn/Common.hpp"

namespace gsbn{

/**
 * \class Random
 * \brief The class generates random number in both CPU and GPU mode.
 */
class Random {

public:

	/**
	 * \fn Random()
	 * \brief Constructor
	 */
	Random();
	/**
	 * \fn ~Random()
	 * \brief Deonstructor
	 */
	~Random();
	
	/**
	 * \fn gen_uniform01_cpu()
	 * \brief Generate float random numbers between 0 and 1 under uniform distribution.
	 * Works in CPU mode.
	 * \param ptr The pointer points to the container which will hold the generated
	 * random numbers.
	 * \param size The size of the container.
	 */
	void gen_uniform01_cpu(float *ptr, size_t size=1);
	/**
	 * \fn gen_normal_cpu()
	 * \brief Generate float random numbers under normal distribution. Works in CPU
	 * mode.
	 * \param ptr The pointer points to the container which will hold the generated
	 * random numbers.
	 * \param size The size of the container.
	 * \param mean The mean value.
	 * \param sigma The standard deviation.
	 */
	void gen_normal_cpu(float *ptr, size_t size=1, float mean=1.0, float sigma=0.0);
	/**
	 * \fn gen_poisson_cpu()
	 * \brief Generate float random numbers under poisson distribution. Works in CPU
	 * mode.
	 * \param ptr The pointer points to the container which will hold the generated
	 * random numbers.
	 * \param size The size of the container.
	 * \param mean The mean value.
	 */
	void gen_poisson_cpu(unsigned int *ptr, size_t size=1, float mean=1.0);
	
	#ifndef CPU_ONLY
	/**
	 * \fn gen_uniform01_gpu()
	 * \brief Generate float random numbers between 0 and 1 under uniform distribution.
	 * Works in GPU mode.
	 * \param ptr The pointer points to the container which will hold the generated
	 * random numbers.
	 * \param size The size of the container.
	 */
	void gen_uniform01_gpu(float *ptr, size_t size=1);
	/**
	 * \fn gen_normal_gpu()
	 * \brief Generate float random numbers under normal distribution. Works in GPU
	 * mode.
	 * \param ptr The pointer points to the container which will hold the generated
	 * random numbers.
	 * \param size The size of the container.
	 * \param mean The mean value.
	 * \param sigma The standard deviation.
	 */
	void gen_normal_gpu(float *ptr, size_t size=1, float mean=1.0, float sigma=0.0);
	/**
	 * \fn gen_poisson_gpu()
	 * \brief Generate float random numbers under poisson distribution. Works in GPU
	 * mode.
	 * \param ptr The pointer points to the container which will hold the generated
	 * random numbers.
	 * \param size The size of the container.
	 * \param mean The mean value.
	 */
	void gen_poisson_gpu(unsigned int *ptr, size_t size=1, float mean=1.0);
	#endif

private:
	default_random_engine *_rng_cpu;
	
	#ifndef CPU_ONLY
	curandGenerator_t *_rng_gpu;
	#endif
	
};

}

#endif // __GSBN_RANDOM_HPP__

