#ifndef __GSBN_J_ARRAY_HPP__
#define __GSBN_J_ARRAY_HPP__

#include "gsbn/Table.hpp"

namespace gsbn{

/**
 * \class JArray
 * \bref The table class that contains j vector for each HCU.
 *
 * The shape of JArray table is [Pj | Ej | Zj | Tj]. 
 */
class JArray : public Table{

public:


	/**
	 * \fn JArray(int mcu_num=0);
	 * \bref A simple constructor of class JArray.
	 * \param mcu_num The number of MCUs.
	 *
	 * The function defines the shape of table. If the \p mcu_num is given, it
	 * will initialize the table with \p mcu_num rows.
	 */
	JArray(int mcu_num=0);
	
	/**
	 * \fn append(int mcu_num);
	 * \bref Append a set of PEZ to JArray table.
	 * \param mcu_num The number of rows added.
	 *
	 * The function append \p mcu_num rows at a time. The rows will be initialized
	 * with 0s.
	 */
	void append(int mcu_num=0);
	
	/**
	 * \fn void update_kernel_cpu(int index, int timestamp, int* spike_container, float kzj, float ke, float kp);
	 * \bref Update a row of JArray table.
	 * \param index The current simulation time.
	 * \param timestamp The current simulation time.
	 * \param kzj The coefficient for Zj.
	 * \param ke The coefficient for E.
	 * \param kp The coefficient for P.
	 *
	 * The function updates the 1 row of the table. The row is indecated by \p
	 * index.
	 */
	void update_kernel_cpu(
		int index, int timestamp,
		float kzj, float ke, float kp
	);
	
	/**
	 * \fn void update_cpu(int index, int timestamp, int* spike_container, float kzj, float ke, float kp);
	 * \bref Update JArray table.
	 * \param timestamp The current simulation time.
	 * \param kzj The coefficient for Zj.
	 * \param ke The coefficient for E.
	 * \param kp The coefficient for P.
	 *
	 * The function updates the all rows of the table.
	 */
	void update_cpu(
		int timestamp,
		float kzj, float ke, float kp
	);

};

}


#endif //__GSBN_JARRAY_HPP__
