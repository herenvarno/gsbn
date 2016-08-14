#include "gsbn/Proj.hpp"

namespace gsbn{

Proj::Proj(int pop_num){
	CHECK_GE(pop_num, 0);
	_pop_num=pop_num;
	
	vector<int> fields={sizeof(int), sizeof(int)};
	int blk_height=100;
	init(fields, blk_height);
}

void Proj::append(int src_pop, int dest_pop){
	CHECK_GE(src_pop, 0);
	CHECK_LT(src_pop, _pop_num);
	CHECK_GE(dest_pop, 0);
	CHECK_LT(dest_pop, _pop_num);
	
	int *ptr = static_cast<int*>(expand(1));
	ptr[0] = src_pop;
	ptr[1] = dest_pop;
}

void Proj::set_pop_num(int pop_num){
	CHECK_GE(pop_num, 0);
	_pop_num=pop_num;
	
	int r = rows();
	for(int i=0;i<r;i++){
		int src_pop = *(static_cast<const int *>(cpu_data(i, 0)));
		int desc_pop = *(static_cast<const int *>(cpu_data(i, 1)));
		if(src_pop >= _pop_num || desc_pop >= _pop_num){
			LOG(WARNING) << "Illegal population number, there are existed projections beyound the range! set pop number failed!";
			break;
		}
	}
}

}
