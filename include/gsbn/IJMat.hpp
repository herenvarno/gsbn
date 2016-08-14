#ifndef __GSBN_IJ_MAT_HPP__
#define __GSBN_IJ_MAT_HPP__

#include "gsbn/Table.hpp"

namespace gsbn{

class IJMat : public Table {

public:

	IJMat();
	
	append();
	
	remove();
	
private:
	
	vector<int> empty_rows;
}

}

#endif //__GSBN_IJ_MAT_HPP__
