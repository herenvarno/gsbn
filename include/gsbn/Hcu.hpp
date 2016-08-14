#ifndef __GHCU_HCU_HPP__
#define __GHCU_HCU_HPP__

#include "Common.hpp"
#include "Table.hpp"
#include "Mcu.hpp"

namespace ghcu{

/**
 * \class Hcu
 * \bref The table class that contains the location/id of the first MCU in each
 * HCU.
 *
 * The shape of Hcu table is [FIRST_MCU_LOCATION]. The \p FIRST_MCU_LOCATION is
 * a integer which indicates the offset.
 * 
 * \warning In current design, the Hcu table should be used and stored
 * in CPU memory.
 */
class Hcu : public Table{

public:

	/**
	 * \fn Hcu::Hcu(int mcu_num, int fanout_num)
	 * \bref A simple constructor of class HCU. With initial data.
	 * \param hcu_num The number of HCUs to be initially added.
	 * \param slot_num The total slots for each HCU.
	 * \param mcu_num The number of mcus in each HCU.
	 * \param fanout_num The number of fanout in each allocated mcus.
	 *
	 * The function not only defines the shape of table, but also fills the table
	 * with \p hcu_num rows, and initializes the available slot for each HCU with
	 * \p slot_num. If the mcu_num is given, The function will atomatically
	 * allocate a Mcu table for each HCU and append MCU to a vector _mcus.
	 */
	Hcu(int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0);

	/**
	 * \fn ~Hcu();
	 * \bref A simple destructor to free allocated Mcu in Hcu table.
	 */
	~Hcu();

	/**
	 * \fn Hcu::append(int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0)
	 * \bref Append new rows to the table and initialize them.
	 * \param hcu_num The number of HCUs to be initially added.
	 * \param slot_num The total slots for each HCU.
	 * \param mcu_num The number of mcus in each HCU.
	 * \param fanout_num The number of fanout in each allocated mcus.
	 *
	 * The function append new rows to the existed HCU table and initialize these
	 * rows. If the mcu_num is given, The function will atomatically
	 * allocate a Mcu table for each HCU and append MCU to a vector _mcus.
	 */
	void append(int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0);
	
	/**
	 * \fn Hcu::add_slot(int index, int slot_num)
	 * \bref A function which modify the available slot of a specific row/HCU.
	 * \param index The index corresponding to the specific row.
	 * \param slot_num The number of slot to be added. It can be any integer,
	 * including positive integer, negative integer as well as zero.
	 * 
	 * The function modifies the slot number of a specific row. Note that, it will
	 * not always "increase" the slot number because the slot_num can be
	 * negative which will make the function acting like Hcu::del_slot.
	 */
	void add_slot(int index, int slot_num);

	/**
	 * \fn Hcu::del_slot(int index, int slot_num)
	 * \bref A function which modify the available fanout of a specific row/HCU.
	 * \param index The index corresponding to the specific row.
	 * \param slot_num The number of slot to be added. It can be any integer,
	 * including positive integer, negative integer as well as zero.
	 *
	 * The function is just a wrap of Hcu::add_slot. The only difference is that
	 * when slot_num is positive, it will "decrease" the slot number.
	 */
	void del_slot(int index, int slot_num);

};

}

#endif //__GHCU_HCU_HPP__
