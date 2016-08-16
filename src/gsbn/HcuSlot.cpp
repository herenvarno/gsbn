#include "gsbn/HcuSlot.hpp"

namespace gsbn{

HcuSlot::HcuSlot(int hcu_num, int slot_num){
	vector<int> fields={sizeof(int)};
	int blk_height=hcu_num>0?hcu_num:100;
	init("hcu_slot", fields, blk_height);
	append(hcu_num, slot_num);
}

void HcuSlot::append(int hcu_num, int slot_num){
	CHECK_GE(hcu_num, 0);
	CHECK_GE(slot_num, 0);
	
	if(hcu_num==0){
		LOG(WARNING) << "Append 0 rows to HCU table, pass!";
		return;
	}
	
	for(int i=0; i<hcu_num; i++){
		void *ptr = expand(1);
		if(ptr){
			int *ptr_0 = static_cast<int*>(ptr);
			*ptr_0 = slot_num;
		}
	}
}

void HcuSlot::add_slot(int index, int slot_num){
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

void HcuSlot::del_slot(int index, int slot_num){
	add_slot(index, -slot_num);
}

}
