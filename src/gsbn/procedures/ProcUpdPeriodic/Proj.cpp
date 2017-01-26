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
	if(_dim_conn > p->_slot_num){
		_dim_conn = p->_slot_num;
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
	_pi0=proj_param.pi0();

	CHECK(_ii = db.create_sync_vector_i32("ii_"+to_string(_id)));
	CHECK(_qi = db.create_sync_vector_i32("qi_"+to_string(_id)));
	CHECK(_di = db.create_sync_vector_i32("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i8(".ssi_"+to_string(_id)));
	CHECK(_pi = db.create_sync_vector_f32("pi_"+to_string(_id)));
	CHECK(_ei = db.create_sync_vector_f32("ei_"+to_string(_id)));
	CHECK(_zi = db.create_sync_vector_f32("zi_"+to_string(_id)));
	CHECK(_pj = db.create_sync_vector_f32("pj_"+to_string(_id)));
	CHECK(_ej = db.create_sync_vector_f32("ej_"+to_string(_id)));
	CHECK(_zj = db.create_sync_vector_f32("zj_"+to_string(_id)));
	CHECK(_pij = db.create_sync_vector_f32("pij_"+to_string(_id)));
	CHECK(_eij = db.create_sync_vector_f32("eij_"+to_string(_id)));
	CHECK(_wij = db.create_sync_vector_f32("wij_"+to_string(_id)));

	CHECK(_si = _ptr_src_pop->_spike);
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	_ii->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn, -1);
	_qi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_di->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_pi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn, _pi0);
	_ei->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_zi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_ssi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_pj->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu, 1.0/_dim_mcu);
	_ej->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_zj->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_pij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu, _pi0/_dim_mcu);
	_eij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_wij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);

	vector<int> list;
	for(int i=0; i<_ptr_dest_pop->_dim_hcu; i++){
		list.push_back(i);
	}
	for(int i=0; i<_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu; i++){
		_ptr_src_pop->_avail_hcu[i].push_back(list);
	}
	_ptr_src_pop->_avail_proj.push_back(_id);
	
	_conn_cnt.resize(_dim_hcu, 0);
	
	// Initially set up connection
	if(proc_param.argf_size()>=1){
		float init_conn_rate=proc_param.argf(0);
		if(init_conn_rate>1.0){
			init_conn_rate = 1.0;
		}
		
		if(init_conn_rate>0.0){
			int conn_per_hcu = int(init_conn_rate * _dim_conn);
			for(int i=0; i<_dim_hcu; i++){
				vector<int> avail_mcu_list(_ptr_src_pop->_dim_mcu);
				std::iota(std::begin(avail_mcu_list), std::end(avail_mcu_list), 0);
				for(int j=0; j<conn_per_hcu && !avail_mcu_list.empty(); j++){
					while(!avail_mcu_list.empty()){
						float random_number;
						_rnd.gen_uniform01_cpu(&random_number);
						int src_mcu_idx = ceil(random_number*(avail_mcu_list.size())-1);
						int src_mcu = avail_mcu_list[src_mcu_idx];
						int *ptr_fanout = _ptr_src_pop->_fanout->mutable_cpu_data();
						if(ptr_fanout[src_mcu]>0){
							ptr_fanout[src_mcu]--;
							_msg->send(_id, src_mcu, i, 1);
							avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
							break;
						}else{
							avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
						}
					}
				}
			}
		}
	}
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
	if(_dim_conn > p->_slot_num){
		_dim_conn = p->_slot_num;
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
	_pi0=proj_param.pi0();

	CHECK(_ii = db.sync_vector_i32("ii_"+to_string(_id)));
	CHECK(_qi = db.sync_vector_i32("qi_"+to_string(_id)));
	CHECK(_di = db.sync_vector_i32("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i8(".ssi_"+to_string(_id)));
	CHECK(_pi = db.sync_vector_f32("pi_"+to_string(_id)));
	CHECK(_ei = db.sync_vector_f32("ei_"+to_string(_id)));
	CHECK(_zi = db.sync_vector_f32("zi_"+to_string(_id)));
	CHECK(_pj = db.sync_vector_f32("pj_"+to_string(_id)));
	CHECK(_ej = db.sync_vector_f32("ej_"+to_string(_id)));
	CHECK(_zj = db.sync_vector_f32("zj_"+to_string(_id)));
	CHECK(_pij = db.sync_vector_f32("pij_"+to_string(_id)));
	CHECK(_eij = db.sync_vector_f32("eij_"+to_string(_id)));
	CHECK(_wij = db.sync_vector_f32("wij_"+to_string(_id)));

	CHECK(_si = _ptr_src_pop->_spike);
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	CHECK_EQ(_ii->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_qi->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_di->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_pi->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_ei->cpu_vector()->size(), _dim_hcu * _dim_conn);
	CHECK_EQ(_zi->cpu_vector()->size(), _dim_hcu * _dim_conn);
	_ssi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	CHECK_EQ(_pj->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_ej->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_zj->cpu_vector()->size(), _dim_hcu * _dim_mcu);
	CHECK_EQ(_pij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_eij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);
	CHECK_EQ(_wij->cpu_vector()->size(), _dim_hcu * _dim_conn * _dim_mcu);

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
	
	// Initially set up connection
	if(proc_param.argf_size()>=1){
		float init_conn_rate=proc_param.argf(0);
		if(init_conn_rate>1.0){
			init_conn_rate = 1.0;
		}
		
		if(init_conn_rate>0.0){
			int conn_per_hcu = int(init_conn_rate * _dim_conn);
			for(int i=0; i<_dim_hcu; i++){
				vector<int> avail_mcu_list(_ptr_src_pop->_dim_mcu);
				std::iota(std::begin(avail_mcu_list), std::end(avail_mcu_list), 0);
				for(int j=0; j<conn_per_hcu && !avail_mcu_list.empty(); j++){
					while(!avail_mcu_list.empty()){
						float random_number;
						_rnd.gen_uniform01_cpu(&random_number);
						int src_mcu_idx = ceil(random_number*(avail_mcu_list.size())-1);
						int src_mcu = avail_mcu_list[src_mcu_idx];
						int *ptr_fanout = _ptr_src_pop->_fanout->mutable_cpu_data();
						if(ptr_fanout[src_mcu]>0){
							ptr_fanout[src_mcu]--;
							_msg->send(_id, src_mcu, i, 1);
							avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
							break;
						}else{
							avail_mcu_list.erase(avail_mcu_list.begin()+src_mcu_idx);
						}
					}
				}
			}
		}
	}
}

void update_ssi_kernel_cpu(
	int i,
	const int *ptr_ii,
	const int *ptr_di,
	const int8_t *ptr_si,
	int *ptr_qi,
	int8_t *ptr_ssi
){
	int ii=ptr_ii[i];
	if(ii>=0){
		int32_t qi = ptr_qi[i];
		qi >>= 1;
		ptr_ssi[i] = (qi & 0x01);
	
		int8_t spk = ptr_si[ii];
		if(spk>0){
			qi |= (0x01 << ptr_di[i]);
		}
		ptr_qi[i]=qi;
	}
}

void update_zep_kernel_cpu(
	int i,
	int j,
	int dim_conn,
	int dim_mcu,
	const int8_t *ptr_ssi,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	const int8_t *ptr_sj,
	float *ptr_pj,
	float *ptr_ej,
	float *ptr_zj,
	float *ptr_pij,
	float *ptr_eij,
	float* ptr_wij,
	float* ptr_epsc,
	float* ptr_bj,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kfti,
	float kftj,
	float wgain,
	float bgain,
	float eps,
	float eps2
){
	int row1 = i;
	int col1 = j;
	int index = i*dim_mcu+j;
	int row2 = i % dim_conn;
	int col2 = i / dim_conn * dim_mcu + j;
	
	float pi = ptr_pi[row1];
	float ei = ptr_ei[row1];
	float zi = ptr_zi[row1];
	float si = ptr_ssi[row1];
	float pj = ptr_pj[col2];
	float ej = ptr_ej[col2];
	float zj = ptr_zj[col2];
	float sj = ptr_sj[col2];
	float pij = ptr_pij[index];
	float eij = ptr_eij[index];
	
	// UPDATE WIJ
	float wij;
	if(kp){
		wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
		ptr_wij[index] = wij;
	}else{
		wij = ptr_wij[index];
	}

	// UPDATE EPSC
	if(si>0){
		ptr_epsc[col2] += wij;
	}
	
	// UPDATE IJ
	pij += (eij - pij)*kp;
	eij += (zi*zj - eij)*ke;
	ptr_pij[index] = pij;
	ptr_eij[index] = eij;

	// UPDATE J
	if(row2==0){
		ptr_epsc[col2] *= (1-kzi);
		if(kp){
			float bj = bgain * log(pj + eps);
			ptr_bj[col2] = bj;
		}
		pj += (ej - pj)*kp;
		ej += (zj - ej)*ke;
		zj *= (1-kzj);
		if(sj>0){
			zj += kftj;
		}
		ptr_pj[col2] = pj;
		ptr_ej[col2] = ej;
		ptr_zj[col2] = zj;
	}
	
	// UPDATE I 
	if(col1==0){
		pi += (ei - pi)*kp;
		ei += (zi - ei)*ke;
		zi *= (1-kzi);
		if(si>0){
			zi += kfti;
		}
		ptr_pi[row1] = pi;
		ptr_ei[row1] = ei;
		ptr_zi[row1] = zi;
	}
}

void Proj::update_ssi_cpu(){
	const int *ptr_ii = _ii->cpu_data();
	const int *ptr_di = _di->cpu_data();
	const int8_t *ptr_si = _si->cpu_data();
	int *ptr_qi = _qi->mutable_cpu_data();
	int8_t *ptr_ssi = _ssi->mutable_cpu_data();

	for(int i=0; i<_dim_conn * _dim_hcu; i++){
		update_ssi_kernel_cpu(
			i,
			ptr_ii,
			ptr_di,
			ptr_si,
			ptr_qi,
			ptr_ssi
		);
	}
}

void Proj::update_epsc_cpu(){
	float *ptr_epsc = _epsc->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	for(int i=0; i<_dim_hcu*_dim_mcu; i++){
		ptr_epsc[i] *= (1-_tauzidt);
	}
}

void Proj::update_zep_cpu(){
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	
	float *ptr_pi = _pi->mutable_cpu_data();
	float *ptr_ei = _ei->mutable_cpu_data();
	float *ptr_zi = _zi->mutable_cpu_data();
	float *ptr_pj = _pj->mutable_cpu_data();
	float *ptr_ej = _ej->mutable_cpu_data();
	float *ptr_zj = _zj->mutable_cpu_data();
	float *ptr_pij = _pij->mutable_cpu_data();
	float *ptr_eij = _eij->mutable_cpu_data();
	float *ptr_wij = _wij->mutable_cpu_data();
	float *ptr_bj = _bj->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_epsc = _epsc->mutable_cpu_data()+ _proj_in_pop * _dim_hcu * _dim_mcu;
	
	const int8_t *ptr_ssi = _ssi->cpu_data();
	const int8_t *ptr_sj = _sj->cpu_data();

	for(int i=0; i<_dim_conn * _dim_hcu; i++){
		for(int j=0; j<_dim_mcu; j++){
			update_zep_kernel_cpu(
				i,
				j,
				_dim_conn,
				_dim_mcu,
				ptr_ssi,
				ptr_pi,
				ptr_ei,
				ptr_zi,
				ptr_sj,
				ptr_pj,
				ptr_ej,
				ptr_zj,
				ptr_pij,
				ptr_eij,
				ptr_wij,
				ptr_epsc,
				ptr_bj,
				_taupdt*prn,
				_tauedt,
				_tauzidt,
				_tauzjdt,
				_kfti,
				_kftj,
				_wgain,
				_bgain,
				_eps,
				_eps2
			);
		}
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
			if((*(_ptr_dest_pop->_slot->mutable_cpu_vector()))[it->dest_hcu]>0){
				(*(_ptr_dest_pop->_slot->mutable_cpu_vector()))[it->dest_hcu]--;
				_msg->send(_id, it->src_mcu, it->dest_hcu, 2);
				add_row(it->src_mcu, it->dest_hcu, it->delay);
			}else{
				_msg->send(_id, it->src_mcu, it->dest_hcu, 3);
			}
			break;
		case 2:
			break;
		case 3:
			(_ptr_src_pop->_fanout->mutable_cpu_data())[it->src_mcu]++;
			break;
		default:
			break;
		}
	}
}


void Proj::add_row(int src_mcu, int dest_hcu, int delay){
	
	if(_conn_cnt[dest_hcu]<_dim_conn){
		int idx = _conn_cnt[dest_hcu];
		_ii->mutable_cpu_data()[dest_hcu*_dim_conn+idx]=src_mcu;
		_di->mutable_cpu_data()[dest_hcu*_dim_conn+idx]=delay;
		_conn_cnt[dest_hcu]++;
	}
}

}
}