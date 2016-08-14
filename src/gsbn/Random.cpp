#include "gsbn/Random.hpp"

namespace gsbn{

int normal_x2_valid = 0;
double normal_x2;
const double Random::gen_normal(double mean, double sigma) {
   // normal distribution with mean m and standard deviation s
   double normal_x1;                   // first random coordinate (normal_x2 is member of class)

   double w;                           // radius
   if (normal_x2_valid) {              // we have a valid result from last call
      normal_x2_valid = 0;
      return normal_x2 * sigma + mean;
   }    
   // make two normally distributed variates by Box-Muller transformation
   do {
      normal_x1 = 2. * gen_uniform01() - 1.;
      normal_x2 = 2. * gen_uniform01() - 1.;
      w = normal_x1*normal_x1 + normal_x2*normal_x2;
   }
   while (w >= 1. || w < 1E-30);
   w = sqrt(log(w)*(-2./w));
   normal_x1 *= w; // normal_x2 *= w;    // normal_x1 and normal_x2 are independent normally distributed variates
   normal_x2_valid = 1;                // save normal_x2 for next call
   return normal_x1 * sigma + mean;
}


const int Random::gen_poisson(double mean) {
	int x = 0;
	double t = 0.0;
	for ( ; ; ) {
		t -= log(gen_uniform01())/mean;
		if (t > 1.0)
			return x;
		x++;
	}
}


const vector<int> Random::select(int N , int n){
    /* Select randomly n numbers from [0 N-1] without write-back */
    if (N<n)
		LOG(FATAL) << "Illegal N<n";
    int k = 0,r;
    vector<char> indi(N,0);
    vector<int> sel(n);
    while (k<n) {
        while (indi[r=rand()%N]>0) ;
        sel[k++] = r;
        indi[r] = 1;
    }
    return sel;
}

}
