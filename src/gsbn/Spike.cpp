#include "gsbn/Spike.hpp"

namespace gsbn{

Spike::Spike(int mcu_num, unsigned char init_spike) : Table::Table(){
	vector<int> fields(1,sizeof(unsigned char));
	int blk_height=mcu_num>0?mcu_num:100;
	init("spike", fields, blk_height);
	append(mcu_num, init_spike);
}

void Spike::append(int mcu_num, unsigned char init_spike) {
	CHECK_GE(mcu_num, 0);
	CHECK_GE(init_spike, 0);
	
	if(mcu_num==0){
		LOG(WARNING) << "Append 0 rows to MCU table, pass!";
		return;
	}
	
	for(int i=0; i<mcu_num; i++){
		
		unsigned char *ptr = static_cast<unsigned char*>(expand(1));
		if(ptr){
			*ptr = init_spike;
		}
	}
}

const string Spike::dump(){

	int r = rows();
	const unsigned char *data_ptr;
	data_ptr=static_cast<const unsigned char*>(cpu_data());	
	ostringstream s;
	
	int i=0;
	while(i<r){
		for(int j=0; j<100 && i<r; j++){
			s << setw(1) << static_cast<unsigned int>(data_ptr[i]);
			i++;
		}
		s << endl;
	}
	std::string str =  s.str();
	return str;
}


}
