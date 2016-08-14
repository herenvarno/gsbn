#ifndef __GSBN_PROJ_HPP__
#define __GSBN_PROJ_HPP__

#include "gsbn/Table.hpp"

namespace gsbn{

/**
 * \class Proj
 * \bref The table class that contains all availale projections the BCPNN
 * network.
 *
 * The shape of Proj table is [SOURCE_POP | DESTINATION_POP]. The id of both
 * population need to be within 0 to maximun population number defined in the
 * Pop table.
 *
 * \warning In current design the Proj table should be used and stored in CPU
 * memory.
 */
class Proj : public Table{

public:

	/**
	 * \fn Proj();
	 * \bref A simple constructor of class Proj. With initial data.
	 * \param pop_num The number of Pop in the BCPNN.
	 *
	 * The function only defines the shape of table, you may consider to use
	 * Proj::append to manually add projections to the table.
	 */
	Proj(int pop_num=0);
	
	/**
	 * \fn append();
	 * \bref Append a projection to Proj table.
	 * \param src_pop The id of source Pop in the BCPNN.
	 * \param dest_pop The id of destination Pop in the BCPNN.
	 *
	 * The function append one projection at a time. The Pop id will be checked in
	 * the function to make sure the population exist.
	 */
	void append(int src_pop, int dest_pop);
	
	/**
	 * \fn set_pop_num()
	 * \bref Set the population number.
	 * \param pop_num The population number in the BCPNN.
	 */
	void set_pop_num(int pop_num);

private:

	int _pop_num;
};

}


#endif //__GHCU_PROJ_HPP__
