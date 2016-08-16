#ifndef __GSBN_HCU_HPP__
#define __GSBN_HCU_HPP__

#include "Table.hpp"

namespace gsbn{

/**
 * \class HcuSlot
 * \bref The table class that contains the available slot of each HCU.
 *
 * The shape of HcuSlot table is [AVAILABLE_SLOT]. The \p AVAILABLE_SLOT is
 * a integer.
 * 
 * \warning In current design, the Hcu table should be used and stored
 * in CPU memory.
 */
class HcuSlot : public Table{

public:

	/**
	 * \fn HcuSlot()
	 * \bref A simple constructor of class HcuSlot. With initial data.
	 * \param hcu_num The number of HCUs to be initially added.
	 * \param slot_num The total slots for each HCU.
	 *
	 * The function not only defines the shape of table, but also fills the table
	 * with \p hcu_num rows, and initializes the available slot for each HCU with
	 * \p slot_num.
	 */
	HcuSlot(int hcu_num=0, int slot_num=0);

	/**
	 * \fn append()
	 * \bref Append new rows to the table and initialize them.
	 * \param hcu_num The number of HCUs to be initially added.
	 * \param slot_num The total slots for each HCU.
	 *
	 * The function append new rows to the existed HcuSlot table and initialize
	 * these rows.
	 */
	void append(int hcu_num=0, int slot_num=0);
	
	/**
	 * \fn add_slot()
	 * \bref A function which modify the available slot of a specific row/HCU.
	 * \param index The index corresponding to the specific row.
	 * \param slot_num The number of slot to be added. It can be any integer,
	 * including positive integer, negative integer as well as zero.
	 * 
	 * The function modifies the slot number of a specific row. Note that, it will
	 * not always "increase" the slot number because the slot_num can be
	 * negative which will make the function acting like del_slot().
	 */
	void add_slot(int index, int slot_num);

	/**
	 * \fn del_slot()
	 * \bref A function which modify the available fanout of a specific row/HCU.
	 * \param index The index corresponding to the specific row.
	 * \param slot_num The number of slot to be added. It can be any integer,
	 * including positive integer, negative integer as well as zero.
	 *
	 * The function is just a wrap of add_slot(). The only difference is that
	 * when slot_num is positive, it will "decrease" the slot number.
	 */
	void del_slot(int index, int slot_num);

};

}

#endif //__GSBN_HCU_HPP__
