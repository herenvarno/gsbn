#ifndef __GSBN_PROC_MAIL_COORDINATE_HPP__
#define __GSBN_PROC_MAIL_COORDINATE_HPP__

#include "gsbn/Database.hpp"

namespace gsbn{
namespace proc_mail{

class Coordinate{
public:
	Coordinate(int idx, int total, vector<int> shape);
	~Coordinate();

	float distance_to(Coordinate c, float d);
	
private:
	float distance_to_center();
	
	vector<int> _shape;
	vector<float> _coor_c;
	vector<int> _coor_0;
	
};


}
}

#endif // __GSBN_PROC_UPD_LAZY_COORDINATE_HPP__
