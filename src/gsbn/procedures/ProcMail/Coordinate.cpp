#include "gsbn/procedures/ProcMail/Coordinate.hpp"

namespace gsbn{
namespace proc_mail{

Coordinate::Coordinate(int idx, int total, vector<int> shape){
	CHECK_GE(idx, 0);
	CHECK_LT(idx, total);
	int dim = shape.size();
	CHECK_GT(dim, 0);
	
	_shape.resize(dim);
	_coor_c.resize(dim);
	_coor_0.resize(dim);
	
	int index=idx;
	for(int i=0; i<dim; i++){
		int d = shape[i];
		CHECK_GT(d, 0);
		_shape[i] = d;
		_coor_c[i] = d * 0.5;
		_coor_0[i] = index / d;
		index %= d;
	}
}

Coordinate::~Coordinate(){
}

float Coordinate::distance_to(Coordinate c, float d){
	int d0 = distance_to_center();
	int d1 = c.distance_to_center();
	return (d + d0 + d1);
}

float Coordinate::distance_to_center(){
	int dim = _shape.size();
	float d2 = 0;
	for(int i=0; i<dim; i++){
		d2 += (_coor_0[i] - _coor_c[i]) * (_coor_0[i] - _coor_c[i]);
	}
	float d=sqrt(d2);
	return d;
}

}
}
