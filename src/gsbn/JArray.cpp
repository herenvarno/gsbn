#include "gsbn/JArray.hpp"

namespace gsbn{

JArray::JArray(int mcu_num){
	vector<int> fields={sizeof(float), sizeof(float), sizeof(float), sizeof(int)};
	int blk_height=mcu_num>0?mcu_num:100;
	init(fields, blk_height);
	append(mcu_num);
}

void JArray::append(int mcu_num){
	CHECK_GE(mcu_num, 0);
	if(mcu_num==0){
		LOG(WARNING) << "Append 0 row to JArray, pass!";
		return;
	}
	expand(mcu_num);
}

void JArray::update_kernel_cpu(
	int index, int timestamp,
	float kzj, float ke, float kp){
	void *ptr = mutable_cpu_data(index);
	if(!ptr){
		LOG(WARNING) << "Cannot get JArray row : index=" << index;
		return;
	}
	
	/*
	 * Update state
	 */
	float pj = static_cast<float *>(ptr)[0];
	float ej = static_cast<float *>(ptr)[1];
	float zj = static_cast<float *>(ptr)[2];
	int tj = static_cast<int *>(ptr+3*sizeof(float))[0];
	
	float pdt = timestamp - tj;
	if (pdt<=0){
		return;
	}

	pj = (pj - ((ej*kp*kzj - ej*ke*kp + ke*kp*zj)/(ke - kp) +
                    (ke*kp*zj)/(kp - kzj))/(ke - kzj))/exp(kp*pdt) +
        ((exp(kp*pdt - ke*pdt)*(ej*kp*kzj - ej*ke*kp + ke*kp*zj))/(ke - kp) +
         (ke*kp*zj*exp(kp*pdt - kzj*pdt))/(kp - kzj))/(exp(kp*pdt)*(ke - kzj));
	ej = (ej - (ke*zj)/(ke - kzj))/exp(ke*pdt) +
        (ke*zj*exp(ke*pdt - kzj*pdt))/(exp(ke*pdt)*(ke - kzj));
	zj = zj*exp(-kzj*pdt);
	tj = timestamp;
	
	static_cast<float *>(ptr)[0] = pj;
	static_cast<float *>(ptr)[1] = ej;
	static_cast<float *>(ptr)[2] = zj;
	static_cast<int *>(ptr+3*sizeof(float))[0] = tj;
	
}

void JArray::update_cpu(
	int timestamp,
	float kzj, float ke, float kp){
		int r=rows();
		for(int i=0;i<r;i++){
			update_kernel_cpu(i, timestamp, kzj, ke, kp);
		}
}

}
