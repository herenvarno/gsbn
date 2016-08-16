#ifndef __GSBN_HCU_HPP__
#define __GSBN_HCU_HPP__

#include "gsbn/Table.hpp"
#include "gsbn/McuFanout.hpp"
#include "gsbn/JArray.hpp"
#include "gsbn/Spike.hpp"

namespace ghcu{

/**
 * \class Hcu
 * \bref The table class that contains the location/id of the first MCU in each
 * HCU.
 *
 * The shape of Hcu table is [FIRST_MCU_INDEX | MUC_NUM]. The \p FIRST_MCU_LOCATION is
 * a integer which indicates the offset.
 * 
 * \warning In current design, the Hcu table should be used and stored
 * in CPU memory.
 */
class Hcu : public Table{

public:

	/**
	 * \fn Hcu()
	 * \bref A simple constructor of class HCU. With initial data.
	 * \param hcu_num The number of HCUs to be initially added.
	 * \param mcu_num The number of mcus in each HCU.
	 * \param fanout_num The number of fanout in each allocated mcus.
	 *
	 * The function not only defines the shape of table, but also fills the table
	 * with \p hcu_num rows, and initializes the available slot for each HCU with
	 * \p slot_num. If the mcu_num is given, The function will atomatically
	 * append MCUs.
	 */
	Hcu(int hcu_num=0, int mcu_num=0, int fanout_num=0);
	
	/**
	 * \fn append()
	 * \bref Append new rows to the table and initialize them.
	 * \param hcu_num The number of HCUs to be initially added.
	 * \param mcu_num The number of mcus in each HCU.
	 * \param fanout_num The number of fanout in each allocated mcus.
	 *
	 * The function append new rows to the existed HCU table and initialize these
	 * rows. If the mcu_num is given, The function will atomatically
	 * append MCUs.
	 */
	void append(int hcu_num=0, int mcu_num=0, int fanout_num=0);
	
	
private:

	static McuFanout mcu_fanout;
	static JArray j_array;
	static Spike spike;

};

McuFanout Hcu.mcu_fanout = mcu_fanout();
JArray Hcu.j_array = j_array();
Spike Hcu.spike = spike();

}

#endif //__GSBN_HCU_HPP__
