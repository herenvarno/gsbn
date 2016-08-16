#include "ghcu/Pop.hpp"

namespace ghcu{

Pop::Pop(int pop_num, int hcu_num, int slot_num, int mcu_num, int fanout_num){
	vector<int> fields={sizeof(Hcu*)};
	int blk_height=hcu_num>0?hcu_num:100;
	init("pop", fields, blk_height);
	append(pop_num, hcu_num, slot_num, mcu_num, fanout_num);
}

Pop::~Pop(){
	int rows = get_rows_cpu();
	for(int i=0; i<rows; i++){
		Hcu **ptr = (Hcu**)(cpu_data(i));
		delete *ptr;
	}
}

void Pop::append(int pop_num, int hcu_num, int slot_num, int mcu_num, int fanout_num){
	CHECK_GE(pop_num, 0);
	CHECK_GE(hcu_num, 0);
	CHECK_GE(slot_num, 0);
	CHECK_GE(mcu_num, 0);
	CHECK_GE(fanout_num, 0);
	
	if(pop_num==0){
		LOG(WARNING) << "Append 0 rows to HCU table, pass!";
		return;
	}
	
	for(int i=0; i<pop_num; i++){
		Hcu **ptr = static_cast<Hcu**>(append_cpu(1));
		if(ptr){
			*ptr = new Hcu(hcu_num, slot_num, mcu_num, fanout_num);
		}
	}
}

}
