#include "gsbn/ConnManager.hpp"

namespace gsbn{

ConnManager::ConnManager(){

}

void ConnManager::init(Database& db){
	CHECK(_j_array = db.table("j_array"));
	CHECK(_i_array = db.table("i_array"));
	CHECK(_ij_mat = db.table("ij_mat"));
	CHECK(_hcu = db.table("hcu"));
	CHECK(_mcu = db.table("mcu"));
	CHECK(_tmp1 = db.table("tmp1"));
	CHECK(_epsc = db.table("epsc"));
	CHECK(_proj = db.table("proj"));
	CHECK(_pop = db.table("pop"));
	CHECK(_mcu_fanout = db.table("mcu_fanout"));
	CHECK(_hcu_slot = db.table("hcu_slot"));
	CHECK(_wij = db.table("wij"));
	CHECK(_tmp2 = db.table("tmp2"));
	CHECK(_tmp3 = db.table("tmp3"));
	CHECK(_addr = db.table("addr"));
	CHECK(_conn = db.table("conn"));
	CHECK(_conn0 = db.table("conn0"));
	CHECK(_hcu_subproj = db.table("hcu_subproj"));
	
	int mcu_num = _mcu->height();
	for(int i=0; i<mcu_num; i++){
		vector<int> l;
		_existed_conn_list.push_back(l);
	}
	
	int conn_num = _conn->height();
	for(int i=0; i<conn_num; i++){
		const int *ptr_conn = static_cast<const int*>(_conn->cpu_data(i));
		_existed_conn_list[ptr_conn[Database::IDX_CONN_SRC_MCU]].push_back(ptr_conn[Database::IDX_CONN_DEST_HCU]);
	}
	
	int conn0_num = _conn->height();
	for(int i=0; i<conn0_num; i++){
		const int *ptr_conn0 = static_cast<const int*>(_conn0->cpu_data(i));
		_existed_conn_list[ptr_conn0[Database::IDX_CONN0_SRC_MCU]].push_back(ptr_conn0[Database::IDX_CONN0_DEST_HCU]);
	}
	
}

void ConnManager::learn(int timestamp, int stim_offset){
	update_phase_1();
	LOG(INFO) << "HERE OK 1";
	update_phase_2(timestamp);
	LOG(INFO) << "HERE OK 2";
	update_phase_3();
	LOG(INFO) << "HERE OK 3";
	update_phase_4();
	LOG(INFO) << "HERE OK 4";
	update_phase_5();
	LOG(INFO) << "HERE OK 5";
	update_phase_6();
	LOG(INFO) << "HERE OK 6";
}

void ConnManager::recall(int timestamp){
/*	update_phase_1();
	update_phase_2(timestamp);
	update_phase_3();
	update_phase_4();
	update_phase_5();
	update_phase_6();*/
}

/*
 * Phase 1: Scan and generate short coming spike list.
 * No need for timestamp
 */
void ConnManager::update_phase_1(){
	int h_conn=_conn->height();
	_tmp2->reset();
	for(int i=0; i<h_conn; i++){
		int *ptr_conn = static_cast<int *>(_conn->mutable_cpu_data(i, 0));
		int queue=ptr_conn[Database::IDX_CONN_QUEUE];
		if(queue & 0x01){
			int *ptr_tmp2 = static_cast<int *>(_tmp2->expand(1));
			ptr_tmp2[Database::IDX_TMP2_CONN]=i;
		}
		ptr_conn[Database::IDX_CONN_QUEUE] = queue >> 1;
	}
}


/*
 * Phase 2: update Wij
 * FIXME : redesign this function
 */
void update_kernel_phase_2_cpu(
	int timestamp,
	int idx_tmp2,
	const void *ptr_tmp2, int w_tmp2,
	void *ptr_i_array, int w_i_array,
	const void *ptr_mcu, int w_mcu,
	const void *ptr_j_array, int w_j_array,
	void *ptr_ij_mat, int w_ij_mat,
	void *ptr_wij, int w_wij,
	float wgain, float eps, float eps2, float kfti,
	float kp, float ke, float kzi, float kzj
	){
	
	const int *ptr_tmp2_data = static_cast<const int *>(ptr_tmp2+idx_tmp2*w_tmp2);
	int idx_wij = ptr_tmp2_data[Database::IDX_TMP2_CONN];
	int idx_mcu = ptr_tmp2_data[Database::IDX_TMP2_SRC_MCU];
	int sub_proj = ptr_tmp2_data[Database::IDX_TMP2_SRC_SUBPROJ];
	float* ptr_wij_data = static_cast<float *>(ptr_wij+idx_wij*w_wij);
	
	void* ptr_ij_mat_data = ptr_ij_mat+idx_wij*w_ij_mat;
	float pij = static_cast<float *>(ptr_ij_mat_data)[Database::IDX_IJ_MAT_PIJ];
	float eij = static_cast<float *>(ptr_ij_mat_data)[Database::IDX_IJ_MAT_EIJ];
	float zi = static_cast<float *>(ptr_ij_mat_data)[Database::IDX_IJ_MAT_ZI2];
	float zj = static_cast<float *>(ptr_ij_mat_data)[Database::IDX_IJ_MAT_ZJ2];
	int tij = static_cast<int *>(ptr_ij_mat_data)[Database::IDX_IJ_MAT_TIJ];
	float pdt = timestamp - tij;
	if(pdt>0)
	{
		static_cast<float *>(ptr_ij_mat_data)[Database::IDX_IJ_MAT_PIJ] =
        (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj)/(ke - kp) -
                (ke*kp*zi*zj)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
        ((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj))/(ke - kp) -
         (ke*kp*zi*zj*exp(kp*pdt - kzi*pdt - kzj*pdt))/
         (kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
    static_cast<float *>(ptr_ij_mat_data)[Database::IDX_IJ_MAT_EIJ] = (eij + (ke*zi*zj)/(kzi - ke + kzj))/exp(ke*pdt) -
        (ke*zi*zj)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
    static_cast<float *>(ptr_ij_mat_data)[Database::IDX_IJ_MAT_ZI2] = zi*exp(-kzi*pdt) + kfti;
    static_cast<float *>(ptr_ij_mat_data)[Database::IDX_IJ_MAT_ZJ2] = zj*exp(-kzj*pdt);
    static_cast<int *>(ptr_ij_mat_data)[Database::IDX_IJ_MAT_TIJ] = timestamp;
	}
	
	void* ptr_i_array_data = static_cast<float *>(ptr_i_array+idx_wij*w_i_array);
	float pi = static_cast<float *>(ptr_i_array_data)[0];
	float ei = static_cast<float *>(ptr_i_array_data)[1];
	zi = static_cast<float *>(ptr_i_array_data)[2];
	int ti = static_cast<int *>(ptr_i_array_data)[3];
	pdt = timestamp - ti;
	if(pdt>0){
		static_cast<float *>(ptr_i_array_data)[0] = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
                    (ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
        ((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
         (ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
		static_cast<float *>(ptr_i_array_data)[1] = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
        (ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
		static_cast<float *>(ptr_i_array_data)[2] = zi*exp(-kzi*pdt) + kfti;
		static_cast<int *>(ptr_i_array_data)[3] = timestamp;
	}
	
	int idx_j_array=static_cast<const int *>(ptr_mcu+idx_mcu*w_mcu)[0];
	float pj = static_cast<const float *>(ptr_j_array+(idx_j_array+sub_proj)*w_j_array)[0];
	ptr_wij_data[0] = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
	
}

void ConnManager::update_phase_2(int timestamp){
	int h_tmp2=_tmp2->height();
	if(h_tmp2<=0)
		return;
	const void *ptr_tmp2 = _tmp2->cpu_data();
	int w_tmp2 = _tmp2->width();
	void *ptr_wij = _wij -> mutable_cpu_data();
	int w_wij = _wij->width();
	void *ptr_i_array = _i_array -> mutable_cpu_data();
	int w_i_array = _i_array->width();
	const void *ptr_mcu = _mcu -> cpu_data();
	int w_mcu = _mcu->width();
	const void *ptr_j_array = _j_array -> cpu_data();
	int w_j_array = _j_array->width();
	void *ptr_ij_mat = _ij_mat -> mutable_cpu_data();
	int w_ij_mat = _ij_mat->width();
	for(int i=0; i<h_tmp2; i++){
		update_kernel_phase_2_cpu(
			timestamp,
			i,
			ptr_tmp2, w_tmp2,
			ptr_i_array, w_i_array,
			ptr_mcu, w_mcu,
			ptr_j_array, w_j_array,
			ptr_ij_mat, w_ij_mat,
			ptr_wij, w_wij,
			_wgain, _eps, _eps2, _kfti,
			_kp, _ke, _kzi, _kzj
		);
	}
}

/*
 * Phase 3: update EPSC
 */
void ConnManager::update_phase_3(){
	int h_tmp2 = _tmp2->height();
	for(int i=0; i<h_tmp2;i++){
		const int *ptr_tmp2_data = static_cast<const int *>(_tmp2->cpu_data(i));
		int idx_mcu = ptr_tmp2_data[1];
		int sub_proj = ptr_tmp2_data[2];
		float *ptr_epsc_data = static_cast<float *>(_epsc->mutable_cpu_data(idx_mcu+sub_proj));
		
		const float* ptr_wij_data = static_cast<const float *>(_wij->cpu_data(i));
		ptr_epsc_data[0] += ptr_wij_data[0];
	}
}

/*
 * Phase 4: deal with special spikes (REQ and ACK)
 */
void ConnManager::update_phase_4(){
	_tmp3->reset();
	
	int h_conn0=_conn0->height();
	for(int i=0; i<h_conn0; i++){
		int *ptr_conn0 = static_cast<int *>(_conn0->mutable_cpu_data(i, 0));
		int queue=ptr_conn0[Database::IDX_CONN0_QUEUE];
		
		if(queue & 0x01){
			int *ptr_hcu_slot;
			int *ptr_conn;
			int *ptr_hcu;
			int idx_mcu_fanout;
			int *ptr_mcu_fanout;
			int idx_hcu;
			int mcu_num;
			int* ptr_tmp3;
			vector<int> *vec;
			vector<int>::iterator position;
			switch(ptr_conn0[Database::IDX_CONN0_TYPE]){
			case 1:	// REQ INCOMMING SPIKE
//				LOG(INFO) << ptr_conn0[Database::IDX_CONN0_DEST_HCU];
				ptr_hcu_slot = static_cast<int *>(_hcu_slot->mutable_cpu_data(ptr_conn0[Database::IDX_CONN0_DEST_HCU]));
				if(ptr_hcu_slot[Database::IDX_HCU_SLOT_VALUE]>0){
					ptr_hcu_slot[Database::IDX_HCU_SLOT_VALUE]--;
					ptr_conn0[Database::IDX_CONN0_TYPE]=2;
				}else{
					ptr_conn0[Database::IDX_CONN0_TYPE]=3;
				}
				queue |= (0x01 << ptr_conn0[Database::IDX_CONN0_DELAY]);
				break;
			case 2:	// ACK INCOMMING SPIKE, ESTABLISH CONNECTION
				ptr_hcu = static_cast<int *>(_hcu->mutable_cpu_data(ptr_conn0[Database::IDX_CONN0_DEST_HCU]));
				mcu_num = ptr_hcu[Database::IDX_HCU_MCU_NUM];
				
				// use tmp3 to initialize new connection
				ptr_tmp3 = static_cast<int*>(_tmp3->expand(1));
				ptr_tmp3[Database::IDX_TMP3_CONN] = _conn->height();
				ptr_tmp3[Database::IDX_TMP3_DEST_HCU] = ptr_conn0[Database::IDX_CONN0_DEST_HCU];
				ptr_tmp3[Database::IDX_TMP3_IJ_MAT_IDX] = _ij_mat->height();
				ptr_tmp3[Database::IDX_TMP3_PI_INIT] = 1.0/mcu_num; // FIXME: The original code use pi0 to initialize it.
				ptr_tmp3[Database::IDX_TMP3_PIJ_INIT] = 1.0/_mcu->height();
			
				ptr_conn = static_cast<int *>(_conn->expand(1));
				ptr_conn[Database::IDX_CONN_SRC_MCU] = ptr_conn0[Database::IDX_CONN0_SRC_MCU];
				ptr_conn[Database::IDX_CONN_DEST_HCU] = ptr_conn0[Database::IDX_CONN0_DEST_HCU];
				ptr_conn[Database::IDX_CONN_SRC_SUBPROJ] = ptr_conn0[Database::IDX_CONN0_SRC_SUBPROJ];
				ptr_conn[Database::IDX_CONN_DELAY] = ptr_conn0[Database::IDX_CONN0_DELAY];
				ptr_conn[Database::IDX_CONN_QUEUE] = 0;
				ptr_conn[Database::IDX_CONN_IJ_MAT_INDEX] = _ij_mat->height();
				MemBlock::type_t t;
				_i_array->expand(1, &t);
				_ij_mat->expand(mcu_num, &t);
				_wij->expand(mcu_num, &t);
				ptr_conn0[Database::IDX_CONN0_TYPE] = 0;
				
				_empty_conn0_list.push_back(i);
				break;
			case 3:	// ACK INCOMMING SPIKE, REFUSE CONNECTION
				ptr_conn0[Database::IDX_CONN0_TYPE] = 0;	//set conn type to EMPTY, connection removed.
				idx_mcu_fanout = ptr_conn0[Database::IDX_CONN0_SRC_MCU];
				idx_hcu = ptr_conn0[Database::IDX_CONN0_DEST_HCU];
				ptr_mcu_fanout = static_cast<int*>(_mcu_fanout->mutable_cpu_data(idx_mcu_fanout));
				ptr_mcu_fanout[Database::IDX_MCU_FANOUT_VALUE]++;	// Recovery the fanout
				// Update the empty row list. It will be reused to establish new connections.
				_empty_conn0_list.push_back(i);
				vec = &(_existed_conn_list[idx_mcu_fanout]);
				position = find(vec->begin(), vec->end(), idx_hcu);
				if (position != vec->end())
					vec->erase(position);
				break;
			default:
				break;
			}
		}
		ptr_conn0[Database::IDX_CONN0_QUEUE] = queue >> 1;
	}
}

/* USELESS 

void update_phase_4(){
	for(vector<special_spike_t>::iterator it=_special_spikes.begin(); it!=_special_spikes.end(); it++){
		if((it->queue)&0x01){
			if(it->type==0){	// REQ
				int *ptr_hcu_slot = static_cast<int *>_hcu_slot->mutable_cpu_data(it->hcu);
				if(ptr_hcu_slot[Database::IDX_HCU_SLOT_NUM]>0){
					ptr_hcu_slot[Database::IDX_HCU_SLOT_NUM]--;
					it->type=1;
				}else{
					it->type=-1;
				}
				it->queue = 0x01 << (it->delay);
			}else if(it->type>0){	// CONNECTION OK
				int *ptr_conn = static_cast<int *>_conn->expand(1);
				ptr_conn[Database::IDX_CONN_SRC_MCU] = it->mcu;
				ptr_conn[Database::IDX_CONN_DEST_HCU] = it->hcu;
				ptr_conn[Database::IDX_CONN_DEST_SUBPROJ] = it->subproj;
				ptr_conn[Database::IDX_CONN_DELAY] = it->delay;
				ptr_conn[Database::IDX_CONN_QUEUE] = 0;
				ptr_conn[Database::IDX_CONN_IJ_MAT_INDEX] = _ij_mat->height();
				MemBlock::type_t t;
				_i_array->expand(1, &t);
				int *ptr_hcu = static_cast<int *>_hcu->mutable_cpu_data(it->hcu);
				_ij_mat->expand(ptr_hcu[Database::IDX_HCU_MCU_NUM], &t);
				_mij->expand(ptr_hcu[Database::IDX_HCU_MCU_NUM], &t);
				_special_spikes.erase(it);
				continue;
			}else{	// CONNECTION REFUSED
				int *ptr_mcu_fanout = static_cast<int *>_mcu_fanout->mutable_cpu_data(it->mcu);
				*ptr_mcu_fanout++;
				_special_spikes.erase(it);
				continue;
			}
		}
		it->queue >>= 1;
	}
}

*/

/*
 * Phase 5: Send special spikes
 */
void ConnManager::update_phase_5(){
	int h_tmp1 = _tmp1->height();
	for(int i=0; i<h_tmp1; i++){
		int idx_mcu = static_cast<const int *>(_tmp1->cpu_data(i))[Database::IDX_TMP1_MCU_IDX];
		int *ptr_mcu_fanout = static_cast<int *>(_mcu_fanout->mutable_cpu_data(idx_mcu));
		if(*ptr_mcu_fanout>0){
			*ptr_mcu_fanout--;
			
			const int *ptr_addr = static_cast<const int *>(_addr->cpu_data(idx_mcu));
			int idx_pop=ptr_addr[Database::IDX_ADDR_POP];
			int idx_hcu=ptr_addr[Database::IDX_ADDR_HCU];

			const int *ptr_hcu = static_cast<const int *>(_hcu->cpu_data(idx_hcu));
			int idx_hcuproj=ptr_hcu[Database::IDX_HCU_SUBPROJ_INDEX];
			int num_hcuproj=ptr_hcu[Database::IDX_HCU_SUBPROJ_NUM];
			vector<int> proj_list;
			bool flag=true;
			for(int j=0; j<num_hcuproj; j++){
				
				int proj_val = static_cast<const int *>(_hcu_subproj->cpu_data(idx_hcuproj+j))[Database::IDX_HCU_SUBPROJ_VALUE];
				if(proj_val<0){
					flag=false;
					break;
				}
				proj_list.push_back(proj_val);
			}
			vector<int> list_available_hcu;
			vector<int> list_available_proj;
			if(flag==true){	// no empty subproj position, get list according to the proj_list
				for(vector<int>::iterator it=proj_list.begin(); it!=proj_list.end(); it++){
					int dest_pop = static_cast<const int *>(_proj->cpu_data(*it))[Database::IDX_PROJ_DEST_POP];
					int iii_hcu = static_cast<const int *>(_pop->cpu_data(dest_pop))[Database::IDX_POP_HCU_INDEX];
					int nnn_hcu = static_cast<const int *>(_pop->cpu_data(dest_pop))[Database::IDX_POP_HCU_NUM];
					for(int k=0;k<nnn_hcu;k++){
						list_available_hcu.push_back(iii_hcu+k);
						list_available_proj.push_back(*it);
					}
				}
			}else{	// there is empty subproj position, all available projections are OK
				int h_proj = _proj->height();
				for(int k=0;k<h_proj;k++){
					int src_pop = static_cast<const int *>(_proj->cpu_data(k))[Database::IDX_PROJ_SRC_POP];
					if(src_pop!=idx_pop){
						continue;
					}
					int dest_pop = static_cast<const int *>(_proj->cpu_data(k))[Database::IDX_PROJ_DEST_POP];
					int iii_hcu = static_cast<const int *>(_pop->cpu_data(dest_pop))[Database::IDX_POP_HCU_INDEX];
					int nnn_hcu = static_cast<const int *>(_pop->cpu_data(dest_pop))[Database::IDX_POP_HCU_NUM];
					for(int l=0;l<nnn_hcu;l++){
						list_available_hcu.push_back(iii_hcu+l);
						list_available_proj.push_back(k);
					}
				}
			}
			vector<int> list=_existed_conn_list[idx_mcu];
			for(vector<int>::iterator it=list.begin(); it!=list.end();it++){
				vector<int>::iterator position = find(list_available_hcu.begin(), list_available_hcu.end(), *it);
				if (position != list_available_hcu.end()){
					list_available_hcu.erase(position);
					list_available_proj.erase(list_available_proj.begin()+distance(list_available_hcu.begin(), position));
				}
			}
			if(list_available_hcu.size()<=0){
				*ptr_mcu_fanout++;
				continue;
			}
			int idx_target_hcu = ceil(Random::gen_uniform01()*list_available_hcu.size()-1);
			int target_hcu = list_available_hcu[idx_target_hcu];
			int target_proj = list_available_proj[idx_target_hcu];
			int target_subproj = 0;
			for(int j=0; j<num_hcuproj; j++){
				int proj_val = static_cast<const int *>(_hcu_subproj->cpu_data(idx_hcuproj+j))[Database::IDX_HCU_SUBPROJ_VALUE];
				if(proj_val==target_proj){
					target_subproj=j;
					break;
				}else if(proj_val<0){
					static_cast<int *>(_hcu_subproj->mutable_cpu_data(idx_hcuproj+j))[Database::IDX_HCU_SUBPROJ_VALUE]=target_proj;
					target_subproj=j;
				}
			}
			int *ptr_conn0;
			if(_empty_conn0_list.empty() || true){
				ptr_conn0 = static_cast<int*>(_conn0->expand(1));
			}else{
				int index = _empty_conn0_list[-1];
				_empty_conn0_list.pop_back();
				ptr_conn0 = static_cast<int*>(_conn0->mutable_cpu_data(index));
			}
			
			ptr_conn0[Database::IDX_CONN0_SRC_MCU] = idx_mcu;
			ptr_conn0[Database::IDX_CONN0_DEST_HCU] = target_hcu;
			ptr_conn0[Database::IDX_CONN0_SRC_SUBPROJ] = target_subproj;
			ptr_conn0[Database::IDX_CONN0_DELAY] = __DELAY__;	// FIXME
			ptr_conn0[Database::IDX_CONN0_QUEUE] = 1 << __DELAY__-1; // FIXME
			ptr_conn0[Database::IDX_CONN0_TYPE] = 1;
			
			_existed_conn_list[idx_mcu].push_back(target_hcu);
		}
	}
}


/*
void ConnManager::create_kernel_cpu(){
	unsigned char spk = static_cast<unsigned char *>(ptr_spk+index*w_spike);
	if(spk!=0){
		int fanout = static_cast<int *>(ptr_mcu_fanout+index*w_mcu_fanout)[0];
		if(fanout>0){
			static_cast<int *>(ptr_mcu_fanout+index*w_mcu_fanout)[0]=fanout-1;
			vector<int> list=conn_list[index];
			
			int mcu_pop = static_cast<int *>(ptr_addr+index*w_addr)[0];
			
			for(int i=0; i<h_proj; i++){
				if(static_cast<int *>(ptr_proj+i*w_proj)[0]==mcu_pop){
					hcu_pop=static_cast<int *>(ptr_proj+i*w_proj)[1];
					int first_hcu = static_cast<int *>(ptr_pop+i*w_pop)[0];
					int num_hcu = static_cast<int *>(ptr_pop+i*w_pop)[1];
					list2 = Random::select(num_hcu, num_hcu);
					for(int j=0; j<list2.size(); j++){
						int tmp = list2[j]+first_hcu;
						for(int k=0; k<list.size(); k++){
							if(tmp==list[k]){
								list2.erase(k);
								break;
							}
						}
						tmp_idx=ceil((Random::gen_uniform01()*list2.size()));
						hcu_idx=list2[tmp_idx];
						list.push_back(hcu_idx);
						
						int *ptr_conn;
						int idx;
						if(!vacum_list.empy()){
							idx=vacum_list.pop();
							ptr_conn = static_cast<int*>(_conn.mutable_cpu_data(idx));
						}else{
							idx=_conn.rows();
							ptr_conn = static_cast<int*>(_conn.append(1));
							_ij_mat.append_gpu(1);
						}
						ptr_conn[0]=index;
						ptr_conn[1]=hcu_idx;
						ptr_conn[2]=1;
						ptr_conn[3]=0x01;
						
						int *ptr_sel=static_cast<int*>(append(1));
						ptr_sel[0]=idx;
						
						return;
					}
				}
			}
		}
	}
}

void ConnManager::create_cpu(){
	r=_spike.rows();
	for(int i=0; i<r; i++){
		create_kernel_cpu(void *ptr, int index, int w);
	}
}
*/

/*
 * Phase 6: increase Zj2
 */
void update_kernel_phase_6_cpu(
	int idx_tmp2,
	const void *ptr_tmp2, int w_tmp2,
	void *ptr_ij_mat, int w_ij_mat,
	float kftj
	){
	
	int idx_ij_mat = static_cast<const int *>(ptr_tmp2+idx_tmp2*w_tmp2)[Database::IDX_TMP2_CONN];
	int *ptr_ij_mat_data = static_cast<int *>(ptr_ij_mat+idx_ij_mat*w_ij_mat);
	ptr_ij_mat_data[Database::IDX_IJ_MAT_ZJ2] += kftj;
}

void ConnManager::update_phase_6(){
	_tmp2->reset();
	int h_tmp1=_tmp1->height();
	for(int i=0; i<h_tmp1; i++){
		int mcu=static_cast<const int *>(_tmp1->cpu_data(i))[Database::IDX_TMP1_MCU_IDX];
		int hcu=static_cast<const int *>(_addr->cpu_data(mcu))[Database::IDX_ADDR_HCU];
		int h_conn=_conn->height();
		for(int j=0; j<h_conn; j++){
			int dest_hcu=static_cast<const int *>(_conn->cpu_data(j))[Database::IDX_CONN_DEST_HCU];
			if(dest_hcu==hcu){
				int ij_mat_first=static_cast<const int *>(_conn->cpu_data(j))[Database::IDX_CONN_IJ_MAT_INDEX];
				int offset = mcu - static_cast<const int *>(_hcu->cpu_data(hcu))[Database::IDX_HCU_MCU_INDEX];
				int ij_mat_idx = ij_mat_first + offset;
				int *ptr = static_cast<int*>(_tmp2->expand(1));
				ptr[Database::IDX_TMP2_CONN]=ij_mat_idx;
			}
		}
	}
	int h_tmp2=_tmp2->height();
	if(h_tmp2<=0)
		return;
	const void *ptr_tmp2 = _tmp2->cpu_data();
	int w_tmp2 = _tmp2->width();
	void *ptr_ij_mat = _ij_mat->mutable_cpu_data();
	int w_ij_mat = _ij_mat->width();
	for(int i=0; i<h_tmp2; i++){
		update_kernel_phase_6_cpu(
			i,
			ptr_tmp2, w_tmp2,
			ptr_ij_mat, w_ij_mat,
			_kftj
		);
	}
}


}
