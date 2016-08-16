#include "gsbn/McuFanout.hpp"

namespace gsbn{

McuFanout::McuFanout(int mcu_num, int fanout_num) : Table::Table(){
	vector<int> fields(1,sizeof(int));
	int blk_height=mcu_num>0?mcu_num:100;
	init("mcu_fanout", fields, blk_height);
	append(mcu_num, fanout_num);
}

void McuFanout::append(int mcu_num, int fanout_num) {
	CHECK_GE(mcu_num, 0);
	CHECK_GE(fanout_num, 0);
	
	if(mcu_num==0){
		LOG(WARNING) << "Append 0 rows to MCU table, pass!";
		return;
	}
	
	for(int i=0; i<mcu_num; i++){
		int *ptr = static_cast<int*>(expand(1));
		if(ptr){
			*ptr = fanout_num;
		}
	}
}

void McuFanout::add_fanout(int index, int fanout_num){
	CHECK_GE(index, 0);
	
	if(fanout_num ==0){
		LOG(WARNING) << "Intend to add 0 fanout, pass!";
		return;
	}
	
	int *ptr = static_cast<int*>(mutable_cpu_data(index, 0));
	if(ptr){
		int tmp = *ptr+fanout_num;
		*ptr = tmp>0?tmp:0;
	}
}

void McuFanout::del_fanout(int index, int fanout_num){
	add_fanout(index, -fanout_num);
}


}
