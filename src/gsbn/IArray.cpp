#include "gsbn/IArray.hpp"

namespace gsbn{

IArray::IArray(){
	vector<int> fields={sizeof(float), sizeof(float), sizeof(float), sizeof(int)};
	int blk_height=100;
	init(fields, blk_height);
}

void IArray::append(int conn_num){
	CHECK_GE(conn_num, 0);
	if(conn_num==0){
		LOG(WARNING) << "Append 0 row to IArray, pass!";
		return;
	}
	expand(conn_num);
}

void IArray::update_kernel_cpu(
	int index, int timestamp,
	float kzi, float ke, float kp){
	void *ptr = mutable_cpu_data(index);
	if(!ptr){
		LOG(WARNING) << "Cannot get IArray row : index=" << index;
		return;
	}
	
	/*
	 * Update state
	 */
	float pi = static_cast<float *>(ptr)[0];
	float ei = static_cast<float *>(ptr)[1];
	float zi = static_cast<float *>(ptr)[2];
	int ti = static_cast<int *>(ptr+3*sizeof(float))[0];
	
	float pdt = timestamp - ti;
	if (pdt<=0){
		return;
	}

	pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
                    (ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
        ((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
         (ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
	ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
        (ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
	zi = zi*exp(-kzi*pdt);
	ti = timestamp;
	
	static_cast<float *>(ptr)[0] = pi;
	static_cast<float *>(ptr)[1] = ei;
	static_cast<float *>(ptr)[2] = zi;
	static_cast<int *>(ptr+3*sizeof(float))[0] = ti;
	
}

void IArray::update_cpu(
	int timestamp,
	float kzj, float ke, float kp){
		int r=rows();
		for(int i=0;i<r;i++){
			update_kernel_cpu(i, timestamp, kzj, ke, kp);
		}
}

}
