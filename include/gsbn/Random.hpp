#ifndef __GSBN_RANDOM_HPP__
#define __GSBN_RANDOM_HPP__

#include "gsbn/Common.hpp"

namespace gsbn{

/**
 * \class Random
 * \bref The table class that contains functions for random number generation.
 * 
 * Usually there is no need to instance this class. All class member functions
 * are defined as static.
 */
class Random {

public:

	/**
	 * \fn void init()
	 * \bref Initialize the Random class.
	 *
	 * The function should be called before other member functions. Otherwise, the
	 * results given by other random function won't be randomly distributed.
	 */
	inline static void init() {
		srand (time(NULL));
	}
	
	/**
	 * \fn const double gen_uniform01()
	 * \bref Generate the random number between 0 and 1 of uniform distribution.
	 * \return The generated number.
	 */
	inline static const double gen_uniform01() {
		return (rand()/(float)RAND_MAX);
	}

	/**
	 * \fn gen_normal(double mean=1.0, double sigma=0.0)
	 * \bref Generate the random number of normal distribution
	 * \param mean The mean value.
	 * \param sigma The deviation value.
	 * \return The generated number.
	 */
	static const double gen_normal(double mean=1.0, double sigma=0.0);

	/**
	 * \fn gen_poisson(double mean=1.0)
	 * \bref Generate the random number of poisson distribution
	 * \param mean The mean value.
	 * \return The generated number.
	 */
	static const int gen_poisson(double mean=1.0);

	/**
	 * \fn select(int N , int n);
	 * \bref Randomly select n integer numbers from [0 N-1].
	 * \param N The upper bound.
	 * \param n The number of returned value.
	 * \return The generated vector.
	 */
	static const vector<int> select(int N , int n);
};

}

#endif // __GSBN_RANDOM_HPP__

