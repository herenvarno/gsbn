#include "gsbn/procedures/ProcUpdMulti/Pop.hpp"

namespace gsbn{
namespace proc_upd_multi{

void Pop::init_new(ProcParam proc_param, PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt, int device_id){
	CHECK_GE(device_id, 0);
	_device_id = device_id;
	
	CHECK(list_pop);
	_list_pop=list_pop;
	
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
	
	float dt;
	CHECK(_glv.getf("dt", dt));
	
	_taumdt = dt/pop_param.taum();
	_wtagain = pop_param.wtagain();
	_maxfqdt = pop_param.maxfq()*dt;
	_igain = pop_param.igain();
	_wgain = pop_param.wgain();
	_lgbias = pop_param.lgbias();
	_snoise = pop_param.snoise();
	
	CHECK(_dsup=db.create_sync_vector_f32("dsup_"+to_string(_id)));
	CHECK(_act=db.create_sync_vector_f32(".act_"+to_string(_id)));
	CHECK(_epsc=db.create_sync_vector_f32("epsc_"+to_string(_id)));
	CHECK(_bj=db.create_sync_vector_f32("bj_"+to_string(_id)));
	CHECK(_spike = db.create_sync_vector_i8("spike_"+to_string(_id)));
	CHECK(_rnd_uniform01 = db.create_sync_vector_f32(".rnd_uniform01_"+to_string(_id)));
	CHECK(_rnd_normal = db.create_sync_vector_f32(".rnd_normal_"+to_string(_id)));
	CHECK(_wmask = db.sync_vector_f32(".wmask"));
	CHECK(_lginp = db.sync_vector_f32(".lginp"));
	
	#ifndef CPU_ONLY
	if(device_id>0){
		_dsup->register_device(device_id, SyncVector<float>::DEVICE);
		_act->register_device(device_id, SyncVector<float>::DEVICE);
		_epsc->register_device(device_id, SyncVector<float>::DEVICE);
		_bj->register_device(device_id, SyncVector<float>::DEVICE);
		_spike->register_device(device_id, SyncVector<int8_t>::DEVICE);
		_rnd_uniform01->register_device(device_id, SyncVector<float>::DEVICE);
		_rnd_normal->register_device(device_id, SyncVector<float>::DEVICE);
		_wmask->register_device(device_id, SyncVector<float>::DEVICE);
		_lginp->register_device(device_id, SyncVector<float>::DEVICE);
	}
	#endif
	
	_dsup->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu, log(1.0/_dim_mcu));
	_act->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu, 1.0/_dim_mcu);
	_spike->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_rnd_uniform01->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_rnd_normal->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_epsc->resize(_dim_hcu * _dim_mcu);
	_bj->resize(_dim_hcu * _dim_mcu);
	
	// External spike for debug
	Parser par(proc_param);
	string filename;
	if(par.args("external spike", filename)){
		_flag_ext_spike=true;
		string line;
		ifstream ext_spk_file (filename);
		if (ext_spk_file.is_open()){
			while(getline(ext_spk_file, line)){
				std::stringstream ss(line);
				std::vector<std::string> vstrings;
				
				while( ss.good() ){
					string substr;
					getline( ss, substr, ',' );
					vstrings.push_back( substr );
				}
				
				int size = vstrings.size();
				if(size>2 && stoi(vstrings[1])==_id && stoi(vstrings[0])>=0){
					vector<int> spike;
					for(int i=2; i<size; i++){
						spike.push_back(stoi(vstrings[i]));
					}
					_ext_spikes[stoi(vstrings[0])]=spike;
				}
			}
			ext_spk_file.close();
		}else{
			LOG(FATAL) << "Unable to open file to load external spikes";
		}
	}else{
		_flag_ext_spike=false;
	}
}

void Pop::init_copy(ProcParam proc_param, PopParam pop_param, Database& db, vector<Pop*>* list_pop, int *hcu_cnt, int *mcu_cnt, int device_id){
	CHECK_GE(device_id, 0);
	_device_id = device_id;
	
	CHECK(list_pop);
	_list_pop=list_pop;
	
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
	
	float dt;
	CHECK(_glv.getf("dt", dt));
	
	_taumdt = dt/pop_param.taum();
	_wtagain = pop_param.wtagain();
	_maxfqdt = pop_param.maxfq()*dt;
	_igain = pop_param.igain();
	_wgain = pop_param.wgain();
	_lgbias = pop_param.lgbias();
	_snoise = pop_param.snoise();
	
	CHECK(_dsup=db.sync_vector_f32("dsup_"+to_string(_id)));
	CHECK(_act=db.create_sync_vector_f32(".act_"+to_string(_id)));
	CHECK(_epsc=db.sync_vector_f32("epsc_"+to_string(_id)));
	CHECK(_bj=db.sync_vector_f32("bj_"+to_string(_id)));
	CHECK(_spike = db.sync_vector_i8("spike_"+to_string(_id)));
	CHECK(_rnd_uniform01 = db.create_sync_vector_f32(".rnd_uniform01_"+to_string(_id)));
	CHECK(_rnd_normal = db.create_sync_vector_f32(".rnd_normal_"+to_string(_id)));
	CHECK(_wmask = db.sync_vector_f32(".wmask"));
	CHECK(_lginp = db.sync_vector_f32(".lginp"));
	
	#ifndef CPU_ONLY
	if(device_id>0){
		_dsup->register_device(device_id, SyncVector<float>::DEVICE);
		_act->register_device(device_id, SyncVector<float>::DEVICE);
		_epsc->register_device(device_id, SyncVector<float>::DEVICE);
		_bj->register_device(device_id, SyncVector<float>::DEVICE);
		_spike->register_device(device_id, SyncVector<int8_t>::DEVICE);
		_rnd_uniform01->register_device(device_id, SyncVector<float>::DEVICE);
		_rnd_normal->register_device(device_id, SyncVector<float>::DEVICE);
		_wmask->register_device(device_id, SyncVector<float>::DEVICE);
		_lginp->register_device(device_id, SyncVector<float>::DEVICE);
	}
	#endif

	CHECK_EQ(_dsup->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	_act->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu, 1.0/_dim_mcu);
	CHECK_EQ(_spike->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	_rnd_uniform01->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_rnd_normal->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	CHECK_EQ(_epsc->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_bj->size(), _dim_hcu * _dim_mcu);
	
	// External spike for debug
	Parser par(proc_param);
	string filename;
	if(par.args("external spike", filename)){
		_flag_ext_spike=true;
		string line;
		ifstream ext_spk_file (filename);
		if (ext_spk_file.is_open()){
			while(getline(ext_spk_file, line)){
				std::stringstream ss(line);
				std::vector<std::string> vstrings;
				
				while( ss.good() ){
					string substr;
					getline( ss, substr, ',' );
					vstrings.push_back( substr );
				}
				
				int size = vstrings.size();
				if(size>2 && stoi(vstrings[1])==_id && stoi(vstrings[0])>=0){
					vector<int> spike;
					for(int i=2; i<size; i++){
						spike.push_back(stoi(vstrings[i]));
					}
					_ext_spikes[stoi(vstrings[0])]=spike;
				}
			}
			ext_spk_file.close();
		}else{
			LOG(FATAL) << "Unable to open file to load external spikes";
		}
	}else{
		_flag_ext_spike=false;
	}
}


void Pop::update_rcv_cpu(){
	float *ptr_epsc = _epsc->mutable_cpu_data();
	float *ptr_bj = _bj->mutable_cpu_data();
	for(int i=0; i<_epsc0.size(); i++){
		const float *ptr_epsc0 = _epsc0[i]->cpu_data();
		const float *ptr_bj0 = _bj0[i]->cpu_data();
		for(int j=0; j<_dim_hcu *_dim_mcu; j++){
			if(i==0){
				ptr_epsc[j] = ptr_epsc0[j];
				ptr_bj[j] = ptr_bj0[j];
			}else{
				ptr_epsc[j] += ptr_epsc0[j];
				ptr_bj[j] += ptr_bj0[j];
			}
		}
	}
}

void Pop::update_snd_cpu(){
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
	const float *ptr_epsc,
	const float *ptr_bj,
	const float *ptr_lginp,
	const float *ptr_wmask,
	const float *ptr_rnd_normal,
	float *ptr_dsup,
	float wgain,
	float lgbias,
	float igain,
	float taumdt
){
	int idx = i*dim_mcu+j;
	float wsup=0;
/*	int offset=0;
	int mcu_num_in_pop = dim_hcu * dim_mcu;
	for(int m=0; m<dim_proj; m++){
		wsup += ptr_bj[offset+idx] + ptr_epsc[offset+idx];
		offset += mcu_num_in_pop;
	}*/
	
	wsup = ptr_bj[idx] + ptr_epsc[idx];
	
	float sup = lgbias + igain * ptr_lginp[idx] + ptr_rnd_normal[idx];
	sup += (wgain * ptr_wmask[i]) * wsup;
	
	float dsup = ptr_dsup[idx];
	ptr_dsup[idx] += (sup - dsup) * taumdt;
}

void update_sup_kernel_2_cpu(
	int i,
	int dim_mcu,
	const float *ptr_dsup,
	float* ptr_act,
	float wtagain
){
	float maxdsup = ptr_dsup[0];
	for(int m=0; m<dim_mcu; m++){
		int idx = i*dim_mcu + m;
		float dsup = ptr_dsup[idx];
		if(dsup>maxdsup){
			maxdsup = dsup;
		}
	}
	float maxact = exp(wtagain*maxdsup);
	float vsum = 0;
	for(int m=0; m<dim_mcu; m++){
		int idx = i*dim_mcu+m;
		float dsup = ptr_dsup[idx];
		float act = exp(wtagain*(dsup-maxdsup));
		if(maxact<1){
			act *= maxact;
		}
		vsum += act;
		ptr_act[idx] = act;
	}

	if(vsum>1){
		for(int m=0; m<dim_mcu; m++){
			int idx = i*dim_mcu + m;
			ptr_act[idx] /= vsum;
		}
	}
}

void update_sup_kernel_3_cpu(
	int i,
	int j,
	int dim_mcu,
	const float *ptr_act,
	const float* ptr_rnd_uniform01,
	int8_t* ptr_spk,
	float maxfqdt
){
	int idx = i*dim_mcu+j;
	ptr_spk[idx] = int8_t(ptr_rnd_uniform01[idx]<ptr_act[idx]*maxfqdt);
}

void Pop::update_sup_cpu(){
	int simstep;
	int lginp_idx;
	int wmask_idx;
	CHECK(_glv.geti("simstep", simstep));
	CHECK(_glv.geti("lginp-idx", lginp_idx));
	CHECK(_glv.geti("wmask-idx", wmask_idx));
	const float* ptr_wmask = _wmask->cpu_data(wmask_idx)+_hcu_start;
	const float* ptr_lginp = _lginp->cpu_data(lginp_idx)+_mcu_start;
	const float* ptr_epsc = _epsc->cpu_data();
	const float* ptr_bj = _bj->cpu_data();
	const float* ptr_rnd_uniform01 = _rnd_uniform01->cpu_data();
	const float* ptr_rnd_normal = _rnd_normal->cpu_data();
	float* ptr_dsup = _dsup->mutable_cpu_data();
	float* ptr_act = _act->mutable_cpu_data();
	int8_t* ptr_spk = _spike->mutable_cpu_data();
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

void Pop::fill_spike(){
	if(!_flag_ext_spike){
		return;
	}
	
	int simstep;
	CHECK(_glv.geti("simstep", simstep));
	vector<int> spk = _ext_spikes[simstep];
	int8_t* ptr_spk = _spike->mutable_cpu_data();
	for(int i=0; i<_dim_hcu*_dim_mcu; i++){
		ptr_spk[i]=0;
	}
	for(int i=0; i<spk.size();i++){
		ptr_spk[spk[i]]=1;
	}
}

}
}
