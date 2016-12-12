#include "gsbn/procedures/ProcHalf/Pop.hpp"

namespace gsbn{
namespace proc_half{

void Pop::init_new(PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt, Msg *msg){
	CHECK(list_pop);
	_list_pop=list_pop;
	CHECK(msg);
	_msg=msg;
	
	_id=_list_pop->size();
	list_pop->push_back(this);
	
	_dim_proj = 0;
	_dim_hcu = pop_param.hcu_num();
	_dim_mcu = pop_param.mcu_num();
	CHECK_GT(_dim_hcu, 0);
	CHECK_GT(_dim_mcu, 0);
	_hcu_start = *hcu_cnt;
	_mcu_start = *mcu_cnt;
	*hcu_cnt += _dim_hcu;
	*mcu_cnt += _dim_hcu * _dim_mcu;
	
	CHECK(_conf=db.table(".conf"));
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt = ptr_conf[Database::IDX_CONF_DT];
	
	_taumdt = dt/pop_param.taum();
	_wtagain = pop_param.wtagain();
	_maxfqdt = pop_param.maxfq()*dt;
	_igain = pop_param.igain();
	_wgain = pop_param.wgain();
	_lgbias = pop_param.lgbias();
	_snoise = pop_param.snoise();
	
	_slot_num = pop_param.slot_num();
	_fanout_num = pop_param.fanout_num();
	
	CHECK(_slot=db.create_sync_vector_i32("slot_"+to_string(_id)));
	CHECK(_fanout=db.create_sync_vector_i32("fanout_"+to_string(_id)));	
	CHECK(_dsup=db.create_sync_vector_f16("dsup_"+to_string(_id)));
	CHECK(_act=db.create_sync_vector_f16(".act_"+to_string(_id)));
	CHECK(_epsc=db.create_sync_vector_f16("epsc_"+to_string(_id)));
	CHECK(_bj=db.create_sync_vector_f16("bj_"+to_string(_id)));
	CHECK(_spike = db.create_sync_vector_i32("spike_"+to_string(_id)));
	CHECK(_rnd_uniform01 = db.create_sync_vector_f32(".rnd_uniform01"+to_string(_id)));
	CHECK(_rnd_normal = db.create_sync_vector_f32(".rnd_normal"+to_string(_id)));
	CHECK(_wmask = db.sync_vector_f32(".wmask"));
	CHECK(_lginp = db.sync_vector_f32(".lginp"));

	_slot->resize(_dim_hcu, _slot_num);
	_fanout->resize(_dim_hcu * _dim_mcu, _fanout_num);
	_dsup->resize(_dim_hcu * _dim_mcu);
	_act->resize(_dim_hcu * _dim_mcu);
	_spike->resize(ceil(float(_dim_hcu * _dim_mcu)/32.0));
	_rnd_uniform01->resize(_dim_hcu * _dim_mcu);
	_rnd_normal->resize(_dim_hcu * _dim_mcu);
	
	_avail_hcu.resize(_dim_hcu * _dim_mcu);
	
}

void Pop::init_copy(PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt, Msg *msg){

	CHECK(list_pop);
	_list_pop=list_pop;
	CHECK(msg);
	_msg=msg;
	
	_id=_list_pop->size();
	list_pop->push_back(this);
	
	_dim_proj = 0;
	_dim_hcu = pop_param.hcu_num();
	_dim_mcu = pop_param.mcu_num();
	CHECK_GT(_dim_hcu, 0);
	CHECK_GT(_dim_mcu, 0);
	_hcu_start = *hcu_cnt;
	_mcu_start = *mcu_cnt;
	*hcu_cnt += _dim_hcu;
	*mcu_cnt += _dim_hcu * _dim_mcu;
	
	CHECK(_conf=db.table(".conf"));
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt = ptr_conf[Database::IDX_CONF_DT];
	
	_taumdt = dt/pop_param.taum();
	_wtagain = pop_param.wtagain();
	_maxfqdt = pop_param.maxfq()*dt;
	_igain = pop_param.igain();
	_wgain = pop_param.wgain();
	_lgbias = pop_param.lgbias();
	_snoise = pop_param.snoise();
	
	_slot_num = pop_param.slot_num();
	_fanout_num = pop_param.fanout_num();
	
	CHECK(_slot=db.sync_vector_i32("slot_"+to_string(_id)));
	CHECK(_fanout=db.sync_vector_i32("fanout_"+to_string(_id)));	
	CHECK(_dsup=db.sync_vector_f16("dsup_"+to_string(_id)));
	CHECK(_act=db.create_sync_vector_f16(".act_"+to_string(_id)));
	CHECK(_epsc=db.sync_vector_f16("epsc_"+to_string(_id)));
	CHECK(_bj=db.sync_vector_f16("bj_"+to_string(_id)));
	CHECK(_spike = db.sync_vector_i32("spike_"+to_string(_id)));
	CHECK(_rnd_uniform01 = db.create_sync_vector_f32(".rnd_uniform01"+to_string(_id)));
	CHECK(_rnd_normal = db.create_sync_vector_f32(".rnd_normal"+to_string(_id)));
	CHECK(_wmask = db.sync_vector_f32(".wmask"));
	CHECK(_lginp = db.sync_vector_f32(".lginp"));

	CHECK_EQ(_slot->size(), _dim_hcu);
	CHECK_EQ(_fanout->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_dsup->size(), _dim_hcu * _dim_mcu);
	_act->resize(_dim_hcu * _dim_mcu);
	CHECK_EQ(_spike->size(), ceil(float(_dim_hcu * _dim_mcu)/32.0));
	_rnd_uniform01->resize(_dim_hcu * _dim_mcu);
	_rnd_normal->resize(_dim_hcu * _dim_mcu);
	
	_avail_hcu.resize(_dim_hcu * _dim_mcu);

}

void Pop::update_rnd_cpu(){
	float *ptr_uniform01= _rnd_uniform01->mutable_cpu_data();
	float *ptr_normal= _rnd_normal->mutable_cpu_data();
	int size = _dim_hcu * _dim_mcu;
	_rnd.gen_uniform01_cpu(ptr_uniform01, size);
	_rnd.gen_normal_cpu(ptr_normal, size, 0, _snoise);
}

void update_sup_kernel_1_cpu(
	int i,
	int j,
	int dim_proj,
	int dim_hcu,
	int dim_mcu,
	const fp16 *ptr_epsc,
	const fp16 *ptr_bj,
	const float *ptr_lginp,
	const float *ptr_wmask,
	const float *ptr_rnd_normal,
	fp16 *ptr_dsup,
	float wgain,
	float lgbias,
	float igain,
	float taumdt
){
	int idx = i*dim_mcu+j;
	float wsup=0;
	int offset=0;
	int mcu_num_in_pop = dim_proj * dim_hcu * dim_mcu;
	for(int m=0; m<dim_proj; m++){
		wsup += fp16_to_fp32(ptr_bj[offset+idx]) + fp16_to_fp32(ptr_epsc[offset+idx]);
		offset += mcu_num_in_pop;
	}
	float sup = lgbias + igain * ptr_lginp[idx] + ptr_rnd_normal[idx];
	sup += (wgain * ptr_wmask[i]) * wsup;
	
	float dsup = fp16_to_fp32(ptr_dsup[idx]);
	float dsup2 = (sup - dsup) * taumdt;
	ptr_dsup[idx] = fp32_to_fp16(dsup + dsup2);
}

void update_sup_kernel_2_cpu(
	int i,
	int dim_mcu,
	const fp16 *ptr_dsup,
	fp16* ptr_act,
	float wtagain	
){
	float maxdsup = fp16_to_fp32(ptr_dsup[0]);
	for(int m=0; m<dim_mcu; m++){
		int idx = i*dim_mcu + m;
		float dsup = fp16_to_fp32(ptr_dsup[idx]);
		if(dsup>maxdsup){
			maxdsup = dsup;
		}
	}
	float maxact = exp(wtagain*maxdsup);
	float vsum = 0;
	for(int m=0; m<dim_mcu; m++){
		int idx = i*dim_mcu+m;
		float dsup = fp16_to_fp32(ptr_dsup[idx]);
		float act = exp(wtagain*(dsup-maxdsup));
		if(maxact<1){
			act *= maxact;
		}
		vsum += act;
		ptr_act[idx] = fp32_to_fp16(act);
	}

	if(vsum>1){
		for(int m=0; m<dim_mcu; m++){
			int idx = i*dim_mcu + m;
			float act = fp16_to_fp32(ptr_act[idx]);
			ptr_act[idx] = fp32_to_fp16(act/vsum);
		}
	}
}

void update_sup_kernel_3_cpu(
	int i,
	int j,
	int dim_mcu,
	const fp16 *ptr_act,
	const float* ptr_rnd_uniform01,
	int* ptr_spk,
	float maxfqdt
){
	int idx = i*dim_mcu+j;
	int i32idx = idx/32;
	int i32offset = idx%32;
	if(ptr_rnd_uniform01[idx]<fp16_to_fp32(ptr_act[idx])*maxfqdt){
		ptr_spk[i32idx] |= 1<<i32offset;
	}else{
		ptr_spk[i32idx] &= ~(1<<i32offset);
	}
}

void Pop::update_sup_cpu(){
	const int* ptr_conf = static_cast<const int*>(_conf->cpu_data());
	int lginp_idx = ptr_conf[Database::IDX_CONF_STIM];
	int wmask_idx = ptr_conf[Database::IDX_CONF_GAIN_MASK];
	const float* ptr_wmask = _wmask->cpu_data(wmask_idx)+_hcu_start;
	const float* ptr_lginp = _lginp->cpu_data(lginp_idx)+_mcu_start;
	const fp16* ptr_epsc = _epsc->cpu_data();
	const fp16* ptr_bj = _bj->cpu_data();
	const float* ptr_rnd_uniform01 = _rnd_uniform01->cpu_data();
	const float* ptr_rnd_normal = _rnd_normal->cpu_data();
	fp16* ptr_dsup = _dsup->mutable_cpu_data();
	fp16* ptr_act = _act->mutable_cpu_data();
	int* ptr_spk = _spike->mutable_cpu_data();
	
	for(int i=0; i<_dim_hcu; i++){
		for(int j=0; j<_dim_mcu; j++){
			update_sup_kernel_1_cpu(
				i,
				j,
				_dim_proj,
				_dim_hcu,
				_dim_mcu,
				ptr_epsc,
				ptr_bj,
				ptr_lginp,
				ptr_wmask,
				ptr_rnd_normal,
				ptr_dsup,
				_wgain,
				_lgbias,
				_igain,
				_taumdt
			);
		}
		update_sup_kernel_2_cpu(
			i,
			_dim_mcu,
			ptr_dsup,
			ptr_act,
			_wtagain
		);
		for(int j=0; j<_dim_mcu; j++){
			update_sup_kernel_3_cpu(
				i,
				j,
				_dim_mcu,
				ptr_act,
				ptr_rnd_uniform01,
				ptr_spk,
				_maxfqdt
			);
		}
	}
}

void Pop::send(){
	const int *ptr_conf = static_cast<const int *>(_conf->cpu_data());
	int plasticity = ptr_conf[Database::IDX_CONF_PLASTICITY];
	if(!plasticity){
		return;
	}
	
	// SEND
	int *ptr_fanout = _fanout->mutable_cpu_data();
	const int *ptr_spike = _spike->cpu_data();
	for(int i=0; i<_dim_hcu * _dim_mcu; i++){
		int size=0;
		for(int j=0; j<_avail_hcu[i].size(); j++){
			size+=_avail_hcu[i][j].size();
		}
		if(ptr_spike[i/32]&(1<<i%32) || ptr_fanout[i]<=0 || size<=0){
			continue;
		}
		ptr_fanout[i]--;
		float random_number;
		_rnd.gen_uniform01_cpu(&random_number);
		int dest_hcu_idx = ceil(random_number*size-1);
		for(int j=0; j<_avail_hcu[i].size(); j++){
			if(dest_hcu_idx < _avail_hcu[i][j].size()){
				_msg->send(_avail_proj[j], i, _avail_hcu[i][j][dest_hcu_idx], 1);
				_avail_hcu[i][j].erase(_avail_hcu[i][j].begin()+dest_hcu_idx);
				break;
			}else{
				dest_hcu_idx -= _avail_hcu[i][j].size();
			}
		}
	}
}

}
}
