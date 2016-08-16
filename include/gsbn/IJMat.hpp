#ifndef __GSBN_IJ_MAT_HPP__
#define __GSBN_IJ_MAT_HPP__

#include "gsbn/Table.hpp"

/**
 * \class IJMat
 * \bref The table class that contains ij matrix for each connection.
 *
 * The shape of IJMat table is [Pij | Eij | Zi2 | Zj2 | Tij]. The table should
 * keep its data in GPU memory.
 */
namespace gsbn{

class IJMat : public Table {

public:
	
	/**
	 * \fn IJMat();
	 * \bref A simple constructor of class IJMat.
	 *
	 * The function defines the shape of table. No initialization is needed.
	 */
	IJMat();
	
	/**
	 * \fn append();
	 * \bref Append a set of PEZT to IJMat table.
	 * \param conn_num The number of rows added.
	 *
	 * The function append \p conn_num rows at a time. The rows will be initialized
	 * with 0s.
	 * 
	 * \warning The function will NOT synchronize the information from GPU memory
	 * to CPU memory because we use GPU expansion here. For more inforamtion about
	 * the expansion type, see expand().
	 */
	void append(int conn_num);

};

}

#endif //__GSBN_IJ_MAT_HPP__
