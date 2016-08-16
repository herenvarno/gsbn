#ifndef __GSBN_POP_HPP__
#define __GSBN_POP_HPP__

#include "Table.hpp"
#include "Hcu.hpp"
#include "HcuSlot.hpp"

namespace gsbn{

/**
 * \class Pop
 * \bref The table class that contains all Populations in the BCPNN network.
 *
 * The shape of Pop table is [FIRST_HCU_INDEX, HCU_NUM]. FIRST_HCU_INDEX gives
 * the first HCU index which belongs to the Pop, and HCU_NUM records the amount
 * of HCUs in the Pop.
 *
 * \warning In current design the Pop table should be used and stored in CPU
 * memory.
 */
class Pop : public Table{

public:

	/**
	 * \fn Pop(int pop_num=0, int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0);
	 * \bref A simple constructor of class Pop. With initial data.
	 * \param pop_num The number of Pop in the BCPNN.
	 * \param hcu_num The number of Hcu in each Pop.
	 * \param slot_num The total slots for each Hcu.
	 * \param mcu_num The number of Mcu in each Hcu.
	 * \param fanout_num The number of fanout in each Mcu.
	 *
	 * The function not only defines the shape of table, but also fills the table
	 * with \p pop_num rows. If the \p hcu_num is given, The function will call
	 * Hcu::append() and HcuSlot::append().
	 */
	Pop(int pop_num=0, int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0);
	
	/**
	 * \fn void append(int pop_num=0, int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0);
	 * \bref Append Pop to Pop table.
	 * \param pop_num The number of Pop in the BCPNN.
	 * \param hcu_num The number of Hcu in each Pop.
	 * \param slot_num The total slots for each Hcu.
	 * \param mcu_num The number of Mcu in each Hcu.
	 * \param fanout_num The number of fanout in each Mcu.
	 *
	 * The function appends the table
	 * with \p pop_num rows. If the \p hcu_num is given, The function will call
	 * Hcu::append() and HcuSlot::append().
	 */
	void append(int pop_num=0, int hcu_num=0, int slot_num=0, int mcu_num=0, int fanout_num=0);

};

private:
	static Hcu hcu;
	static HcuSlot hcu_slot;
	
}



#endif //__GHCU_POP_HPP__
