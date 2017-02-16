#include "gsbn/procedures/ProcUpdPeriodic/Proj.hpp"

namespace gsbn{
namespace proc_upd_periodic{

void Proj::init_new(ProcParam proc_param, ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, Msg *msg){

	CHECK(list_pop);
	CHECK(list_proj);
	CHECK(msg);

	_list_pop = list_pop;
	_list_proj = list_proj;
	_msg = msg;

	_id = _list_proj->size();
	_list_proj->push_back(this);

	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];

	Pop* p= _ptr_dest_pop;
	_dim_hcu = p->_dim_hcu;
	_dim_mcu = p->_dim_mcu;
	_dim_conn = (_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
	_slot_num=proj_param.slot_num();
	if(_dim_conn > _slot_num){
		_dim_conn = _slot_num;
	}
	_proj_in_pop = p->_dim_proj;
	p->_dim_proj++;
	p->_epsc->mutable_cpu_vector()->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	p->_bj->mutable_cpu_vector()->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	_epsc = p->_epsc;
	_bj = p->_bj;

	CHECK(_conf=db.table(".conf"));
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt = ptr_conf[Database::IDX_CONF_DT];

	_tauzidt=dt/proj_param.tauzi();
	_tauzjdt=dt/proj_param.tauzj();
	_tauedt=dt/proj_param.taue();
	_taupdt=dt/proj_param.taup();
	_eps=dt/proj_param.taup();
	_eps2=(dt/proj_param.taup())*(dt/proj_param.taup());
	_kfti=1/(proj_param.maxfq() * proj_param.tauzi());
	_kftj=1/(proj_param.maxfq() * proj_param.tauzj());
	_bgain=proj_param.bgain();
	_wgain=proj_param.wgain();

	CHECK(_ii = db.create_sync_vector_i32("ii_"+to_string(_id)));
	CHECK(_qi = db.create_sync_vector_i32("qi_"+to_string(_id)));
	CHECK(_di = db.create_sync_vector_i32("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i32(".ssi_"+to_string(_id)));
	CHECK(_siq = db.create_sync_vector_i8(".siq_"+to_string(_id)));
	CHECK(_pi = db.create_sync_vector_f32("pi_"+to_string(_id)));
	CHECK(_ei = db.create_sync_vector_f32("ei_"+to_string(_id)));
	CHECK(_zi = db.create_sync_vector_f32("zi_"+to_string(_id)));
	CHECK(_ti = db.create_sync_vector_i32("ti_"+to_string(_id)));
	CHECK(_ssj = db.create_sync_vector_i32(".ssj_"+to_string(_id)));
	CHECK(_pj = db.create_sync_vector_f32("pj_"+to_string(_id)));
	CHECK(_ej = db.create_sync_vector_f32("ej_"+to_string(_id)));
	CHECK(_zj = db.create_sync_vector_f32("zj_"+to_string(_id)));
	CHECK(_pij = db.create_sync_vector_f32("pij_"+to_string(_id)));
	CHECK(_eij = db.create_sync_vector_f32("eij_"+to_string(_id)));
	CHECK(_zi2 = db.create_sync_vector_f32("zi2_"+to_string(_id)));
	CHECK(_zj2 = db.create_sync_vector_f32("zj2_"+to_string(_id)));
	CHECK(_tij = db.create_sync_vector_i32("tij_"+to_string(_id)));
	CHECK(_wij = db.create_sync_vector_f32("wij_"+to_string(_id)));
	CHECK(_slot=db.create_sync_vector_i32("slot_"+to_string(_id)));

	CHECK(_si = _ptr_src_pop->_spike);
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	_ii->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn, -1);
	_qi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_di->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_pi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn, 1.0/_dim_conn);
	_ei->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_zi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_ti->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_siq->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_pj->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu, 1.0/_dim_mcu);
	_ej->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_zj->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_pij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu, 1.0/_dim_conn/_dim_mcu);
	_eij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_zi2->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_zj2->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_tij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_wij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_slot->mutable_cpu_vector()->resize(_dim_hcu, _slot_num);

	vector<int> list;
	for(int i=0; i<_ptr_dest_pop->_dim_hcu; i++){
		list.push_back(i);
	}
	for(int i=0; i<_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu; i++){
		_ptr_src_pop->_avail_hcu[i].push_back(list);
	}
	_ptr_src_pop->_avail_proj.push_back(_id);
	_ptr_src_pop->_avail_proj_hcu_start.push_back(_ptr_dest_pop->_hcu_start);
	
	_conn_cnt.resize(_dim_hcu, 0);
	
	/*
	// Initially set up connection
	if(proc_param.argf_size()>=1){
		float init_conn_rate=proc_param.argf(0);
		if(init_conn_rate>1.0){
			init_conn_rate = 1.0;
		}
		
		if(init_conn_rate>0.0){
			int conn_per_hcu = int(init_conn_rate * _dim_conn);
			for(int i=0; i<_dim_hcu; i++){
				vector<int> avail_mcu_list(_ptr_src_pop->_dim_hcu*_dim_mcu);
				std::iota(std::begin(avail_mcu_list), std::end(avail_mcu_list), 0);
				for(int j=0; j<conn_per_hcu && !avail_mcu_list.empty(); j++){
					while(!avail_mcu_list.empty()){
						float random_number;
						_rnd.gen_uniform01_cpu(&random_number);
						int src_mcu_idx = ceil(random_number*(avail_mcu_list.size())-1);
						int src_mcu = avail_mcu_list[src_mcu_idx];
						if(_ptr_src_pop->validate_conn(src_mcu, _id, i)){
							int *ptr_fanout = _ptr_src_pop->_fanout->mutable_cpu_data();
							ptr_fanout[src_mcu]--;
							LOG(INFO) << "src=" << src_mcu << " proj=" << _id << " dest=" << i;
							_ptr_src_pop->update_avail_hcu(src_mcu, _id, i, true);
							(*(_slot->mutable_cpu_vector()))[i]--;
							add_row(src_mcu, i, 1);
							
							avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
							break;
						}
						avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
					}
				}
			}
		}
	}
	*/
}

void Proj::init_copy(ProcParam proc_param, ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, Msg *msg){

	CHECK(list_pop);
	CHECK(list_proj);
	CHECK(msg);

	_list_pop = list_pop;
	_list_proj = list_proj;
	_msg = msg;

	_id = _list_proj->size();
	_list_proj->push_back(this);

	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];

	Pop* p= _ptr_dest_pop;
	_dim_hcu = p->_dim_hcu;
	_dim_mcu = p->_dim_mcu;
	_dim_conn = (_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
	_slot_num=proj_param.slot_num();
	if(_dim_conn > _slot_num){
		_dim_conn = _slot_num;
	}
	_proj_in_pop = p->_dim_proj;
	p->_dim_proj++;
	CHECK_LE(p->_epsc->cpu_vector()->size(), p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	CHECK_LE(p->_bj->mutable_cpu_vector()->size(), p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	_epsc = p->_epsc;
	_bj = p->_bj;

	CHECK(_conf=db.table(".conf"));
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt = ptr_conf[Database::IDX_CONF_DT];

	_tauzidt=dt/proj_param.tauzi();
	_tauzjdt=dt/proj_param.tauzj();
	_tauedt=dt/proj_param.taue();
	_taupdt=dt/proj_param.taup();
	_eps=dt/proj_param.taup();
	_eps2=(dt/proj_param.taup())*(dt/proj_param.taup());
	_kfti=1/(proj_param.maxfq() * proj_param.tauzi());
	_kftj=1/(proj_param.maxfq() * proj_param.tauzj());
	_bgain=proj_param.bgain();
	_wgain=proj_param.wgain();

	CHECK(_ii = db.sync_vector_i32("ii_"+to_string(_id)));
	CHECK(_qi = db.sync_vector_i32("qi_"+to_string(_id)));
	CHECK(_di = db.sync_vector_i32("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i32(".ssi_"+to_string(_id)));
	CHECK(_siq = db.create_sync_vector_i8(".siq_"+to_string(_id)));
	CHECK(_pi = db.sync_vector_f32("pi_"+to_string(_id)));
	CHECK(_ei = db.sync_vector_f32("ei_"+to_string(_id)));
	CHECK(_zi = db.sync_vector_f32("zi_"+to_string(_id)));
	CHECK(_ti = db.sync_vector_i32("ti_"+to_string(_id)));
	CHECK(_ssj = db.create_sync_vector_i32(".ssj_"+to_string(_id)));
	CHECK(_pj = db.sync_vector_f32("pj_"+to_string(_id)));
	CHECK(_ej = db.sync_vector_f32("ej_"+to_string(_id)));
	CHECK(_zj = db.sync_vector_f32("zj_"+to_string(_id)));
	CHECK(_pij = db.sync_vector_f32("pij_"+to_string(_id)));
	CHECK(_eij = db.sync_vector_f32("eij_"+to_string(_id)));
	CHECK(_zi2 = db.sync_vector_f32("zi2_"+to_string(_id)));
	CHECK(_zj2 = db.sync_vector_f32("zj2_"+to_string(_id)));
	CHECK(_tij = db.sync_vector_i32("tij_"+to_string(_id)));
	CHECK(_wij = db.sync_vector_f32("wij_"+to_string(_id)));
	CHECK(_slot = db.sync_vector_i32("slot_"+to_string(_id)));

	CHECK(_si = _ptr_src_pop->_spike);
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	CHECK_EQ(_ii->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_qi->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_di->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_pi->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_ei->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_zi->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_ti->cpu_vector()->size(), _dim_hcu * _dim_conn);
	_siq->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	CHECK_EQ(_pj->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_ej->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_zj->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_pij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_eij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_zi2->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_zj2->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_tij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_wij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_slot->cpu_vector()->size(), _dim_hcu);
	
	const int *ptr_ii = _ii->cpu_data();
	_conn_cnt.resize(_dim_hcu);
	for(int i=0; i<_dim_hcu; i++){
		for(int j=0; j<_dim_conn; j++){
			if(ptr_ii[i*_dim_conn+j]<0){
				_conn_cnt[i]=j;
				break;
			}
			_conn_cnt[i]=_dim_conn;
		}
	}
	
	vector<int> list;
	for(int i=0; i<_ptr_dest_pop->_dim_hcu; i++){
		list.push_back(i);
	}
	for(int i=0; i<_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu; i++){
		vector<int> list_cpy=list;
		for(int x=0; x<_dim_hcu; x++){
			for(int y=0; y<_conn_cnt[x]; y++){
				int mcu = ptr_ii[x*_dim_conn+y];
				if(mcu==i){
					for(int z=0; z<list_cpy.size(); z++){
						if(list_cpy[z]==x){
							list_cpy.erase(list_cpy.begin()+z);
							break;
						}
					}
					break;
				}
			}
		}
		_ptr_src_pop->_avail_hcu[i].push_back(list_cpy);
	}
	
	_ptr_src_pop->_avail_proj.push_back(_id);
	_ptr_src_pop->_avail_proj_hcu_start.push_back(_ptr_dest_pop->_hcu_start);
	
	/*
	// Initially set up connection
	if(proc_param.argf_size()>=1){
		float init_conn_rate=proc_param.argf(0);
		if(init_conn_rate>1.0){
			init_conn_rate = 1.0;
		}
		
		if(init_conn_rate>0.0){
			int conn_per_hcu = int(init_conn_rate * _dim_conn);
			for(int i=0; i<_dim_hcu; i++){
				vector<int> avail_mcu_list(_ptr_src_pop->_dim_hcu*_dim_mcu);
				std::iota(std::begin(avail_mcu_list), std::end(avail_mcu_list), 0);
				for(int j=0; j<conn_per_hcu && !avail_mcu_list.empty(); j++){
					while(!avail_mcu_list.empty()){
						float random_number;
						_rnd.gen_uniform01_cpu(&random_number);
						int src_mcu_idx = ceil(random_number*(avail_mcu_list.size())-1);
						int src_mcu = avail_mcu_list[src_mcu_idx];
						if(_ptr_src_pop->validate_conn(src_mcu, _id, i)){
							int *ptr_fanout = _ptr_src_pop->_fanout->mutable_cpu_data();
							ptr_fanout[src_mcu]--;
							_ptr_src_pop->update_avail_hcu(src_mcu, _id, i, true);
							(*(_slot->mutable_cpu_vector()))[i]--;
							add_row(src_mcu, i, 1);
							avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
							break;
						}
						avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
					}
				}
			}
		}
	}
	*/
}

void update_siq_kernel_cpu(
	int i,
	const int *ptr_ii,
	const int *ptr_di,
	const int8_t *ptr_si,
	int *ptr_qi,
	int8_t *ptr_siq
){
	int ii=ptr_ii[i];
	if(ii>=0){
		int32_t qi = ptr_qi[i];
		qi >>= 1;
		ptr_siq[i] = (qi & 0x01);
	
		int8_t spk = ptr_si[ii];
		if(spk>0){
			qi |= (0x01 << ptr_di[i]);
		}
		ptr_qi[i]=qi;
	}
}

void update_ij_kernel_cpu(
	int i,
	int j,
	int dim_conn,
	int dim_mcu,
	float *ptr_pi,
	float *ptr_zi,
	float *ptr_pj,
	float *ptr_zj,
	float *ptr_pij,
	float *ptr_eij,
	float* ptr_wij,
	float kp,
	float ke,
	float wgain,
	float eps,
	float eps2
){
	int offset_i = i*dim_conn;
	int offset_j = i*dim_mcu+j;
	int offset_ij = i*dim_conn*dim_mcu+j;
	float *ptr_pi_0 = ptr_pi+offset_i;
	float *ptr_zi_0 = ptr_zi+offset_i;
	float *ptr_pj_0 = ptr_pj+offset_j;
	float *ptr_zj_0 = ptr_zj+offset_j;
	float *ptr_pij_0 = ptr_pij+	offset_ij;
	float *ptr_eij_0 = ptr_eij+	offset_ij;
	float *ptr_wij_0 = ptr_wij+	offset_ij;

	float pj = *ptr_pj_0;
	float zj = *ptr_zj_0;
	for(int xx=0; xx<dim_conn; xx++){
		float pi = ptr_pi_0[xx];
		float zi = ptr_zi_0[xx];
		float pij = *ptr_pij_0;
		float eij = *ptr_eij_0;
	
		// UPDATE WIJ
		float wij;
		if(kp){
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			*ptr_wij_0 = wij;
		}
	
		// UPDATE IJ
		pij += (eij - pij)*kp;
		eij += (zi*zj - eij)*ke;
		*ptr_pij_0 = pij;
		*ptr_eij_0 = eij;

		ptr_pij_0 += dim_mcu;
		ptr_eij_0 += dim_mcu;
		ptr_wij_0 += dim_mcu;
	}
}

void update_j_kernel_cpu(
	int index,
	const int8_t *ptr_sj,
	float *ptr_pj,
	float *ptr_ej,
	float *ptr_zj,
	float* ptr_bj,
	float kp,
	float ke,
	float kzj,
	float kftj,
	float bgain,
	float eps
){
	float pj = ptr_pj[index];
	float ej = ptr_ej[index];
	float zj = ptr_zj[index];
	float sj = ptr_sj[index];
	// UPDATE J
	if(kp){
		float bj = bgain * log(pj + eps);
		ptr_bj[index] = bj;
	}
	pj += (ej - pj)*kp;
	ej += (zj - ej)*ke;
	zj *= (1-kzj);
	if(sj>0){
		zj += kftj;
	}
	ptr_pj[index] = pj;
	ptr_ej[index] = ej;
	ptr_zj[index] = zj;
}

void update_i_kernel_cpu(
	int index,
	const int8_t *ptr_ssi,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	float kp,
	float ke,
	float kzi,
	float kfti
){
	float pi = ptr_pi[index];
	float ei = ptr_ei[index];
	float zi = ptr_zi[index];
	float si = ptr_ssi[index];
	// UPDATE I 
	pi += (ei - pi)*kp;
	ei += (zi - ei)*ke;
	zi *= (1-kzi);
	if(si>0){
		zi += kfti;
	}
	ptr_pi[index] = pi;
	ptr_ei[index] = ei;
	ptr_zi[index] = zi;
}

void Proj::update_siq_cpu(){
	const int *ptr_ii = _ii->cpu_data();
	const int *ptr_di = _di->cpu_data();
	const int8_t *ptr_si = _si->cpu_data();
	int *ptr_qi = _qi->mutable_cpu_data();
	int8_t *ptr_siq = _siq->mutable_cpu_data();

	for(int i=0; i<_dim_conn * _dim_hcu; i++){
		update_siq_kernel_cpu(
			i,
			ptr_ii,
			ptr_di,
			ptr_si,
			ptr_qi,
			ptr_siq
		);
	}
}

void update_epsc_kernel_cpu(
	int index,
	int dim_conn,
	int dim_mcu,
	const float *ptr_wij,
	const int8_t *ptr_siq,
	float *ptr_epsc,
	float kzi
){
	int offset_wij = (index/dim_mcu)*dim_conn*dim_mcu+index%dim_mcu;
	int offset_siq = (index/dim_mcu)*dim_conn;
//	cout << index << "," << offset_wij << "," << offset_ssi << endl;
	const float *ptr_wij_0 = ptr_wij+offset_wij;
	const int8_t *ptr_siq_0 = ptr_siq+offset_siq;
	//FIXME
	float epsc = ptr_epsc[index] * (1-(0.001/0.05));
	for(int i=0; i<dim_conn; i++){
		if(*ptr_siq_0>0){
			epsc += *ptr_wij_0;
		}
		ptr_wij_0 += dim_mcu;
		ptr_siq_0 += 1;
	}
	ptr_epsc[index] = epsc;
}

void Proj::update_epsc_cpu(){
	float *ptr_epsc = _epsc->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	const float *ptr_wij = _wij->cpu_data();
	const int8_t *ptr_siq = _siq->cpu_data();
	for(int i=0; i<_dim_hcu*_dim_mcu; i++){
		update_epsc_kernel_cpu(
			i,
			_dim_conn,
			_dim_mcu,
			ptr_wij,
			ptr_siq,
			ptr_epsc,
			_tauzidt
		);
	}/*
	cout << "epsc = "<< endl;
	for(int i=0;i<_dim_hcu*_dim_mcu; i++){
		cout << ptr_epsc[i]	<< ",";
	}
	cout << endl;*/
}

void Proj::update_ij_cpu(){
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	
	float *ptr_pi = _pi->mutable_cpu_data();
	float *ptr_zi = _zi->mutable_cpu_data();
	float *ptr_pj = _pj->mutable_cpu_data();
	float *ptr_zj = _zj->mutable_cpu_data();
	float *ptr_pij = _pij->mutable_cpu_data();
	float *ptr_eij = _eij->mutable_cpu_data();
	float *ptr_wij = _wij->mutable_cpu_data();
	
	const int8_t *ptr_siq = _siq->cpu_data();
	const int8_t *ptr_sj = _sj->cpu_data();

	for(int i=0; i<_dim_hcu; i++){
		for(int j=0; j<_dim_mcu; j++){
			update_ij_kernel_cpu(
				i,
				j,
				_dim_conn,
				_dim_mcu,
				ptr_pi,
				ptr_zi,
				ptr_pj,
				ptr_zj,
				ptr_pij,
				ptr_eij,
				ptr_wij,
				_taupdt*prn,
				_tauedt,
				_wgain,
				_eps,
				_eps2
			);
		}
	}
}

void Proj::update_j_cpu(){
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	float *ptr_pj = _pj->mutable_cpu_data();
	float *ptr_ej = _ej->mutable_cpu_data();
	float *ptr_zj = _zj->mutable_cpu_data();
	float *ptr_bj = _bj->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	const int8_t *ptr_sj = _sj->cpu_data();

	for(int i=0; i<_dim_mcu * _dim_hcu; i++){
			update_j_kernel_cpu(
				i,
				ptr_sj,
				ptr_pj,
				ptr_ej,
				ptr_zj,
				ptr_bj,
				_taupdt*prn,
				_tauedt,
				_tauzjdt,
				_kftj,
				_bgain,
				_eps
			);
		}
}

void Proj::update_i_cpu(){
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	
	float *ptr_pi = _pi->mutable_cpu_data();
	float *ptr_ei = _ei->mutable_cpu_data();
	float *ptr_zi = _zi->mutable_cpu_data();
	const int8_t *ptr_siq = _siq->cpu_data();

	for(int i=0; i<_dim_conn * _dim_hcu; i++){
			update_i_kernel_cpu(
				i,
				ptr_siq,
				ptr_pi,
				ptr_ei,
				ptr_zi,
				_taupdt*prn,
				_tauedt,
				_tauzidt,
				_kfti
			);
		}
}

void Proj::receive_spike(){
	// RECEIVE
	const int *ptr_conf = static_cast<const int *>(_conf->cpu_data());
	int plasticity = ptr_conf[Database::IDX_CONF_PLASTICITY];
	if(!plasticity){
		return;
	}
	
	vector<msg_t> list_msg = _msg->receive(_id);
	for(vector<msg_t>::iterator it = list_msg.begin(); it!=list_msg.end(); it++){
		if(it->dest_hcu < 0 ||
			it->dest_hcu > _ptr_dest_pop->_dim_hcu ||
			it->src_mcu < 0 ||
			it->src_mcu > _ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu){
			continue;
		}
		switch(it->type){
		case 1:
			if((*(_slot->mutable_cpu_vector()))[it->dest_hcu]>0){
				(*(_slot->mutable_cpu_vector()))[it->dest_hcu]--;
				_msg->send(_id, it->src_mcu, it->dest_hcu, 2);
				add_row(it->src_mcu, it->dest_hcu, it->delay);
			}else{
				_msg->send(_id, it->src_mcu, it->dest_hcu, 3);
			}
			break;
		case 2:
			_ptr_src_pop->update_avail_hcu(it->src_mcu, _id, it->dest_hcu, true);
			break;
		case 3:
			_ptr_src_pop->update_avail_hcu(it->src_mcu, _id, it->dest_hcu, false);
			(_ptr_src_pop->_fanout->mutable_cpu_data())[it->src_mcu]++;
			break;
		default:
			break;
		}
	}
	/*
	for(int i=0; i<_ii->cpu_vector()->size(); i++){
		cout << (*(_ii->cpu_vector()))[i] << ",";
	}
	cout << endl;*/
}


void Proj::add_row(int src_mcu, int dest_hcu, int delay){
	
	if(_conn_cnt[dest_hcu]<_dim_conn){
		int idx = _conn_cnt[dest_hcu];
		_ii->mutable_cpu_data()[dest_hcu*_dim_conn+idx]=src_mcu;
		_di->mutable_cpu_data()[dest_hcu*_dim_conn+idx]=delay;
		_conn_cnt[dest_hcu]++;
	}
}

void Proj::init_conn(ProcParam proc_param){
	// Initially set up connection
	Parser par(proc_param);
	float init_conn_rate;
	if(par.argf("init conn rate", init_conn_rate)){
		if(init_conn_rate>1.0){
			init_conn_rate = 1.0;
		}
		
		if(init_conn_rate>0.0){
			int conn_per_hcu = int(init_conn_rate * _dim_conn);
			for(int i=0; i<_dim_hcu; i++){
				vector<int> avail_mcu_list(_ptr_src_pop->_dim_hcu*_dim_mcu);
				std::iota(std::begin(avail_mcu_list), std::end(avail_mcu_list), 0);
				for(int j=0; j<conn_per_hcu && !avail_mcu_list.empty(); j++){
					while(!avail_mcu_list.empty()){
						float random_number;
						_rnd.gen_uniform01_cpu(&random_number);
						int src_mcu_idx = ceil(random_number*(avail_mcu_list.size())-1);
						int src_mcu = avail_mcu_list[src_mcu_idx];
						if(_ptr_src_pop->validate_conn(src_mcu, _id, i)){
							int *ptr_fanout = _ptr_src_pop->_fanout->mutable_cpu_data();
							ptr_fanout[src_mcu]--;
							_ptr_src_pop->update_avail_hcu(src_mcu, _id, i, true);
							(*(_slot->mutable_cpu_vector()))[i]--;
							add_row(src_mcu, i, 1);
							
							avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
							break;
						}
						avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
					}
				}
			}
		}
	}
}

}
}
