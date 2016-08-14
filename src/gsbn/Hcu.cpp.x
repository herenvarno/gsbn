#include "ghcu/Hcu.hpp"

namespace ghcu{

Hcu::Hcu(int hcu_num, int slot_num, int mcu_num, int fanout_num){
	vector<int> fields={sizeof(int), sizeof(Mcu*)};
	int blk_height=hcu_num>0?hcu_num:100;
	init(fields, blk_height);
	append(hcu_num, slot_num, mcu_num, fanout_num);
}

Hcu::~Hcu(){
	int rows = get_rows_cpu();
	for(int i=0; i<rows; i++){
		Mcu **ptr = (Mcu**)(cpu_data(i,1));
		delete *ptr;
	}
}

void Hcu::append(int hcu_num, int slot_num, int mcu_num, int fanout_num){
	CHECK_GE(hcu_num, 0);
	CHECK_GE(slot_num, 0);
	CHECK_GE(mcu_num, 0);
	CHECK_GE(fanout_num, 0);
	
	if(hcu_num==0){
		LOG(WARNING) << "Append 0 rows to HCU table, pass!";
		return;
	}
	
	for(int i=0; i<hcu_num; i++){
		void *ptr = append_cpu(1);
		if(ptr){
			int *ptr_0 = static_cast<int*>(ptr);
			Mcu **ptr_1 = static_cast<Mcu**>(ptr+sizeof(int));
			*ptr_0 = slot_num;
			*ptr_1 = new Mcu(mcu_num, fanout_num);;
		}
	}
}

void Hcu::add_slot(int index, int slot_num){
	CHECK_GE(index, 0);
	
	if(slot_num==0){
		LOG(WARNING) << "Add 0 slot to HCU, pass!";
		return;
	}
	
	int *ptr = static_cast<int*>(mutable_cpu_data(index, 0));
	if(ptr){
		int tmp = *ptr+slot_num;
		*ptr = tmp>0?tmp:0;
	}
}

void Hcu::del_slot(int index, int slot_num){
	add_slot(index, -slot_num);
}

}
