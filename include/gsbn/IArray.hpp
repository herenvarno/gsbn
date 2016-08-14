#ifndef __GSBN_I_ARRAY_HPP__
#define __GSBN_I_ARRAY_HPP__

#include "gsbn/Common.hpp"
#include "gsbn/Table.hpp"

namespace gsbn{

/**
 * \class IArray
 * \bref The table class that contains i vector for each connection.
 *
 * The shape of IArray table is [Pi | Ei | Zi | Ti]. 
 */
class IArray : public Table{

public:

	/**
	 * \fn IArray();
	 * \bref A simple constructor of class IArray.
	 *
	 * The function defines the shape of table. No initialization is needed.
	 */
	IArray();
	
	/**
	 * \fn append();
	 * \bref Append a set of PEZ to IArray table.
	 * \param conn_num The number of rows added.
	 *
	 * The function append \p conn_num rows at a time. The rows will be initialized
	 * with 0s.
	 */
	void append(int conn_num=0);
	
	/**
	 * \fn update_kernel_cpu();
	 * \bref Update a row of JArray table.
	 * \param index The current simulation time.
	 * \param timestamp The current simulation time.
	 * \param kzi The coefficient for Zi.
	 * \param ke The coefficient for E.
	 * \param kp The coefficient for P.
	 *
	 * The function updates the 1 row of the table. The row is indecated by \p
	 * index.
	 */
	void update_kernel_cpu(
		int index, int timestamp,
		float kzi, float ke, float kp
	);
	
	/**
	 * \fn update_cpu();
	 * \bref Update JArray table.
	 * \param timestamp The current simulation time.
	 * \param kzi The coefficient for Zi.
	 * \param ke The coefficient for E.
	 * \param kp The coefficient for P.
	 *
	 * The function updates the all rows of the table.
	 */
	void update_cpu(
		int timestamp,
		float kzi, float ke, float kp
	);

};

}


#endif //__GSBN_I_ARRAY_HPP__
