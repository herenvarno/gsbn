#include "gsbn/procedures/ProcNetBatch/Proj.hpp"

namespace gsbn{
namespace proc_net_batch{

void Proj::init_new(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, Msg *msg){

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

	CHECK(_ii = db.create_sync_vector_i("ii_"+to_string(_id)));
	CHECK(_qi = db.create_sync_vector_i("qi_"+to_string(_id)));
	CHECK(_di = db.create_sync_vector_i("di_"+to_string(_id)));
	CHECK(_ssi = db.create_sync_vector_i(".ssi_"+to_string(_id)));
	CHECK(_pi = db.create_sync_vector_f("pi_"+to_string(_id)));
	CHECK(_ei = db.create_sync_vector_f("ei_"+to_string(_id)));
	CHECK(_zi = db.create_sync_vector_f("zi_"+to_string(_id)));
	CHECK(_ti = db.create_sync_vector_i("ti_"+to_string(_id)));
	CHECK(_ssj = db.create_sync_vector_i(".ssj_"+to_string(_id)));
	CHECK(_pj = db.create_sync_vector_f("pj_"+to_string(_id)));
	CHECK(_ej = db.create_sync_vector_f("ej_"+to_string(_id)));
	CHECK(_zj = db.create_sync_vector_f("zj_"+to_string(_id)));
	CHECK(_pij = db.create_sync_vector_f("pij_"+to_string(_id)));
	CHECK(_eij = db.create_sync_vector_f("eij_"+to_string(_id)));
	CHECK(_zi2 = db.create_sync_vector_f("zi2_"+to_string(_id)));
	CHECK(_zj2 = db.create_sync_vector_f("zj2_"+to_string(_id)));
	CHECK(_tij = db.create_sync_vector_i("tij_"+to_string(_id)));
	CHECK(_wij = db.create_sync_vector_f("wij_"+to_string(_id)));

	CHECK(_si = _ptr_src_pop->_spike);
	CHECK(_sj = _ptr_dest_pop->_spike);
	
	_ii->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn, -1);
	_qi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_di->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_pi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn, _pi0);
	_ei->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_zi->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_ti->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn);
	_pj->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu, 1.0/_dim_mcu);
	_ej->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_zj->mutable_cpu_vector()->resize(_dim_hcu * _dim_mcu);
	_pij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu, _pi0/_dim_mcu);
	_eij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_zi2->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_zj2->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_tij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);
	_wij->mutable_cpu_vector()->resize(_dim_hcu * _dim_conn * _dim_mcu);

	vector<int> list;
	for(int i=0; i<_ptr_dest_pop->_dim_hcu; i++){
		list.push_back(i);
	}

	_avail_hcu.resize(_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
	for(int i=0; i<_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu; i++){
		_avail_hcu[i].insert(_avail_hcu[i].end(), list.begin(), list.end());
	}
	
	_conn_cnt.resize(_dim_hcu, 0);

}

void Proj::init_copy(ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop, Msg *msg){
}

void update_full_kernel_cpu(
	int i,
	int j,
	int dim_conn,
	int dim_mcu,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	float *ptr_wij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float wgain,
	float eps,
	float eps2
){
	if(j==0){
		float pi = ptr_pi[i];
		float zi = ptr_zi[i];
		int ti = ptr_ti[i];
		int pdt = simstep - ti;
		if(pdt<=0){
			ptr_ti[i]=simstep;
		}else{
			float ei = ptr_ei[i];
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
				(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
				((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
				(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
				(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			zi = zi*exp(-kzi*pdt);
			ti = simstep;
		
			ptr_pi[i] = pi;
			ptr_ei[i] = ei;
			ptr_zi[i] = zi;
			ptr_ti[i] = ti;
		}
	}
	
	int index = i*dim_mcu+j;
	
	float pij = ptr_pij[index];
	int tij = ptr_tij[index];
	float zi2 = ptr_zi2[index];
	int pdt = simstep - tij;
	if(pdt<=0){
		ptr_tij[index]=simstep;
	}else{
		float pij = ptr_pij[index];
		float eij = ptr_eij[index];
		float zj2 = ptr_zj2[index];
	
		pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
			(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
			((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
			(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
			(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
		eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
			(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
		zi2 = zi2*exp(-kzi*pdt);
		zj2 = zj2*exp(-kzj*pdt);
		tij = simstep;
			 	
		ptr_pij[index] = pij;
		ptr_eij[index] = eij;
		ptr_zi2[index] = zi2;
		ptr_zj2[index] = zj2;
		ptr_tij[index] = tij;
			
		// update wij and epsc
		float wij;
		if(kp){
			float pi = ptr_pi[i];
			float pj = ptr_pj[i/dim_conn*dim_mcu + j];
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
		}
	}
}

void update_j_kernel_cpu(
	int idx,
	const int *ptr_sj,
	float *ptr_pj,
	float *ptr_ej,
	float *ptr_zj,
	float *ptr_bj,
	float *ptr_epsc,
	float kp,
	float ke,
	float kzj,
	float kzi,
	float kftj,
	float bgain,
	float eps
){
	float pj = ptr_pj[idx];
	float ej = ptr_ej[idx];
	float zj = ptr_zj[idx];
	int sj = ptr_sj[idx];
	
/*	if(idx%10==0){
		LOG(INFO) << "epsc: " << ptr_epsc[idx];
	}*/
	ptr_epsc[idx] *= (1-kzi);
	
	if(kp){
		float bj = bgain * log(pj + eps);
		ptr_bj[idx] = bj;
	}
	
	pj += (ej - pj)*kp;
	ej += (zj - ej)*ke;
	zj *= (1-kzj);
	if(sj>0){
		zj += kftj;
	}

	ptr_pj[idx] = pj;
	ptr_ej[idx] = ej;
	ptr_zj[idx] = zj;
}

void update_row_kernel_cpu(
	int i,
	int j,
	int dim_conn,
	int dim_mcu,
	const int *ptr_ssi,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	float* ptr_wij,
	float* ptr_epsc,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kfti,
	float wgain,
	float eps,
	float eps2
){
	int row = ptr_ssi[i];
	int col = j;
	int index = row*dim_mcu+col;
	
	if(j==0){
		float pi = ptr_pi[row];
		float ei = ptr_ei[row];
		float zi = ptr_zi[row];
		int ti = ptr_ti[row];
		int pdt = simstep - ti;
		if(pdt<=0){
			ptr_zi[row] += kfti;
			ptr_ti[row] = simstep;
		}else{
			float pi = ptr_pi[row];
			float ei = ptr_ei[row];
		
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
				(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
				((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
				(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
				(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			zi = zi*exp(-kzi*pdt) + kfti;
			ti = simstep;
			ptr_pi[row] = pi;
			ptr_ei[row] = ei;
			ptr_zi[row] = zi;
			ptr_ti[row] = ti;
		}
	}
	
	float pij = ptr_pij[index];
	int tij = ptr_tij[index];
	float zi2 = ptr_zi2[index];
	int pdt = simstep - tij;
	if(pdt<=0){
		ptr_zi2[index] += kfti;
		ptr_tij[index] = simstep;
	}else{
		float eij = ptr_eij[index];
		float zj2 = ptr_zj2[index];
	
		pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
			(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
			((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
			(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
			(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
		eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
			(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
		zi2 = zi2*exp(-kzi*pdt)+kfti;
		zj2 = zj2*exp(-kzj*pdt);
		tij = simstep;
			 	
		ptr_pij[index] = pij;
		ptr_eij[index] = eij;
		ptr_zi2[index] = zi2;
		ptr_zj2[index] = zj2;
		ptr_tij[index] = tij;
		
		float wij;
		int idx_hcu = row / dim_conn;
		int idx_mcu = idx_hcu * dim_mcu + j;
		if(kp){
			float pi = ptr_pi[row];
			float pj = ptr_pj[idx_mcu];
/*		if(j==0){
				LOG(INFO) << "["<< i << ", " << j<<"]: pi=("<<i<<")" << pi << ", pj("<<i/dim_conn*dim_mcu + j<<")=" << pj << ", pij("<<index<<")=" << pij;
			}*/
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
		}else{
			wij = ptr_wij[index];
		}
		ptr_epsc[idx_mcu] += wij;
	}
}

void update_col_kernel_cpu(
	int i,
	int j,
	int dim_mcu,
	const int *ptr_ii,
	const int *ptr_ssj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kftj
){
	int row = i;
	if(ptr_ii[i]<0){
		return;
	}
	int col = ptr_ssj[j];
	int index = row*dim_mcu+col;
	
	int tij = ptr_tij[index];
	float zj2 = ptr_zj2[index];
	int pdt = simstep - tij;
	if(pdt<=0){
		zj2 += kftj;
		ptr_zj2[index]=zj2;
		ptr_tij[index]=simstep;
	}else{
		float pij = ptr_pij[index];
		float eij = ptr_eij[index];
		float zi2 = ptr_zi2[index];
	
		pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
			(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
			((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
			(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
			(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
		eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
			(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
		zi2 = zi2*exp(-kzi*pdt);
		zj2 = zj2*exp(-kzj*pdt)+kftj;
		tij = simstep;
			 	
		ptr_pij[index] = pij;
		ptr_eij[index] = eij;
		ptr_zi2[index] = zi2;
		ptr_zj2[index] = zj2;
		ptr_tij[index] = tij;
	}
}





void Proj::update_full_cpu(){
	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	float old_prn = ptr_conf1[Database::IDX_CONF_OLD_PRN];
	if(old_prn!=prn){
		float *ptr_pi = _pi->mutable_cpu_data();
		float *ptr_ei = _ei->mutable_cpu_data();
		float *ptr_zi = _zi->mutable_cpu_data();
		int *ptr_ti = _ti->mutable_cpu_data();
		const float *ptr_pj = _pj->cpu_data();
		float *ptr_pij = _pij->mutable_cpu_data();
		float *ptr_eij = _eij->mutable_cpu_data();
		float *ptr_zi2 = _zi2->mutable_cpu_data();
		float *ptr_zj2 = _zj2->mutable_cpu_data();
		int *ptr_tij = _tij->mutable_cpu_data();
		float *ptr_wij = _wij->mutable_cpu_data();

		for(int i=0; i<_dim_hcu * _dim_conn; i++){
			for(int j=0; j<_dim_mcu; j++){
				update_full_kernel_cpu(
					i,
					j,
					_dim_conn,
					_dim_mcu,
					ptr_pi,
					ptr_ei,
					ptr_zi,
					ptr_ti,
					ptr_pj,
					ptr_pij,
					ptr_eij,
					ptr_zi2,
					ptr_zj2,
					ptr_tij,
					ptr_wij,
					simstep,
					_taupdt*old_prn,
					_tauedt,
					_tauzidt,
					_tauzjdt,
					_wgain,
					_eps,
					_eps2
				);
			}
		}
	}
}

void Proj::update_j_cpu(){
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf[Database::IDX_CONF_PRN];
	float *ptr_pj = _pj->mutable_cpu_data();
	float *ptr_ej = _ej->mutable_cpu_data();
	float *ptr_zj = _zj->mutable_cpu_data();
	float *ptr_epsc = _epsc->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_bj = _bj->mutable_cpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	const int *ptr_sj = _sj->mutable_cpu_data();

	for(int i=0; i<_dim_hcu * _dim_mcu; i++){
		update_j_kernel_cpu(
			i,
			ptr_sj,
			ptr_pj,
			ptr_ej,
			ptr_zj,
			ptr_bj,
			ptr_epsc,
			_taupdt*prn,
			_tauedt,
			_tauzjdt,
			_tauzidt,
			_kftj,
			_bgain,
			_eps
		);
	}
}

void Proj::update_ss_cpu(){
	// get active in spike
	CONST_HOST_VECTOR(int, *v_si) = _si->cpu_vector();
	CONST_HOST_VECTOR(int, *v_ii) = _ii->cpu_vector();
	CONST_HOST_VECTOR(int, *v_di) = _di->cpu_vector();
	HOST_VECTOR(int, *v_qi) = _qi->mutable_cpu_vector();
	HOST_VECTOR(int, *v_ssi) = _ssi->mutable_cpu_vector();
	
	v_ssi->clear();
	for(int i=0; i<_dim_conn * _dim_hcu; i++){
		if((*v_ii)[i]<0){
			continue;
		}
		(*v_qi)[i] >>= 1;
		if((*v_qi)[i] & 0x01){
			v_ssi->push_back(i);
		}
	
		int spk = (*v_si)[(*v_ii)[i]];
		if(spk>0){
			(*v_qi)[i] |= (0x01 << (*v_di)[i]);
		}
	}
	
	// get active out spike
	CONST_HOST_VECTOR(int, *v_sj) = _sj->cpu_vector();
	HOST_VECTOR(int, *v_ssj) = _ssj->mutable_cpu_vector();
	v_ssj->clear();
	for(int i=0; i<_dim_mcu; i++){
		if((*v_sj)[i]){
			v_ssj->push_back(i);
		}
	}
	
	LOG(INFO) << "ssi:" << v_ssi->size() << " , ssj:" << v_ssj->size();
}

void Proj::update_row_cpu(){
	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	
	float *ptr_pi = _pi->mutable_cpu_data();
	float *ptr_ei = _ei->mutable_cpu_data();
	float *ptr_zi = _zi->mutable_cpu_data();
	int *ptr_ti = _ti->mutable_cpu_data();
	const float *ptr_pj = _pj->cpu_data();
	float *ptr_pij = _pij->mutable_cpu_data();
	float *ptr_eij = _eij->mutable_cpu_data();
	float *ptr_zi2 = _zi2->mutable_cpu_data();
	float *ptr_zj2 = _zj2->mutable_cpu_data();
	int *ptr_tij = _tij->mutable_cpu_data();
	float *ptr_wij = _wij->mutable_cpu_data();
	float *ptr_epsc = _epsc->mutable_cpu_data()+ _proj_in_pop * _dim_hcu * _dim_mcu;
	
	const int *ptr_ssi = _ssi->cpu_data();
	int active_row_num = _ssi->cpu_vector()->size();

	for(int i=0; i<active_row_num; i++){
		for(int j=0; j<_dim_mcu; j++){
			update_row_kernel_cpu(
				i,
				j,
				_dim_conn,
				_dim_mcu,
				ptr_ssi,
				ptr_pi,
				ptr_ei,
				ptr_zi,
				ptr_ti,
				ptr_pj,
				ptr_pij,
				ptr_eij,
				ptr_zi2,
				ptr_zj2,
				ptr_tij,
				ptr_wij,
				ptr_epsc,
				simstep,
				_taupdt*prn,
				_tauedt,
				_tauzidt,
				_tauzjdt,
				_kfti,
				_wgain,
				_eps,
				_eps2
			);
		}
	}
}

void Proj::update_col_cpu(){
	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	
	float *ptr_pij = _pij->mutable_cpu_data();
	float *ptr_eij = _eij->mutable_cpu_data();
	float *ptr_zi2 = _zi2->mutable_cpu_data();
	float *ptr_zj2 = _zj2->mutable_cpu_data();
	int *ptr_tij = _tij->mutable_cpu_data();
	
	const int *ptr_ii = _ii->cpu_data();
	const int *ptr_ssj = _ssj->cpu_data();
	int active_col_num = _ssj->cpu_vector()->size();
	for(int i=0; i<_dim_hcu * _dim_conn; i++){
		for(int j=0; j<active_col_num; j++){
			update_col_kernel_cpu(
				i,
				j,
				_dim_mcu,
				ptr_ii,
				ptr_ssj,
				ptr_pij,
				ptr_eij,
				ptr_zi2,
				ptr_zj2,
				ptr_tij,
				simstep,
				_taupdt * prn,
				_tauedt,
				_tauzidt,
				_tauzjdt,
				_kftj
			);
		}
	}
}

void Proj::send_receive_cpu(){
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
				add_row_cpu(it->src_mcu, it->dest_hcu, it->delay);
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
	
	// SEND
	for(int i=0; i<_dim_hcu * _dim_mcu; i++){
		int *ptr_fanout = _ptr_src_pop->_fanout->mutable_cpu_data();
		const int *ptr_spike = _ptr_src_pop->_spike->cpu_data();
		if(ptr_spike[i]<=0 || ptr_fanout[i]<=0 || _avail_hcu[i].size()<=0){
			continue;
		}
		ptr_fanout[i]--;
		float random_number;
		_rnd.gen_uniform01_cpu(&random_number);
		int dest_hcu_idx = ceil(random_number*_avail_hcu[i].size()-1);
		_msg->send(_id, i, _avail_hcu[i][dest_hcu_idx], 1);
		_avail_hcu[i].erase(_avail_hcu[i].begin()+dest_hcu_idx);
	}
}


void Proj::add_row_cpu(int src_mcu, int dest_hcu, int delay){
	if(_conn_cnt[dest_hcu]<_dim_conn){
		int idx = _conn_cnt[dest_hcu];
		_ii->mutable_cpu_data()[dest_hcu*_dim_conn+idx]=src_mcu;
		_di->mutable_cpu_data()[dest_hcu*_dim_conn+idx]=delay;
		_conn_cnt[dest_hcu]++;
	}
}

}
}