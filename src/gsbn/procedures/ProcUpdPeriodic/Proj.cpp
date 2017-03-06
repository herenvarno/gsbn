#include "gsbn/procedures/ProcUpdPeriodic/Proj.hpp"

namespace gsbn{
namespace proc_upd_periodic{

void Proj::init_new(ProcParam proc_param, ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop){

	CHECK(list_pop);
	CHECK(list_proj);
	_list_pop = list_pop;
	_list_proj = list_proj;

	_id = _list_proj->size();
	_list_proj->push_back(this);

	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];

	Pop* p= _ptr_dest_pop;
	_dim_hcu = p->_dim_hcu;
	_dim_mcu = p->_dim_mcu;
	_dim_conn = (_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
	int slot_num=proj_param.slot_num();
	if(_dim_conn > slot_num){
		_dim_conn = slot_num;
	}
	_proj_in_pop = p->_dim_proj;
	p->_dim_proj++;
	p->_epsc->mutable_cpu_vector()->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	p->_bj->mutable_cpu_vector()->resize(p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	_epsc = p->_epsc;
	_bj = p->_bj;

	float dt;
	CHECK(_glv.getf("dt", dt));

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
	if(proj_param.has_tauepsc()){
		_tauepscdt = dt/proj_param.tauepsc();
	}else{
		_tauepscdt = _tauzidt;
	}

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
}

void Proj::init_copy(ProcParam proc_param, ProjParam proj_param, Database& db, vector<Proj*>* list_proj, vector<Pop*>* list_pop){

	CHECK(list_pop);
	CHECK(list_proj);
	_list_pop = list_pop;
	_list_proj = list_proj;

	_id = _list_proj->size();
	_list_proj->push_back(this);

	_ptr_src_pop = (*list_pop)[proj_param.src_pop()];
	_ptr_dest_pop = (*list_pop)[proj_param.dest_pop()];

	Pop* p= _ptr_dest_pop;
	_dim_hcu = p->_dim_hcu;
	_dim_mcu = p->_dim_mcu;
	_dim_conn = (_ptr_src_pop->_dim_hcu * _ptr_src_pop->_dim_mcu);
	int slot_num=proj_param.slot_num();
	if(_dim_conn > slot_num){
		_dim_conn = slot_num;
	}
	_proj_in_pop = p->_dim_proj;
	p->_dim_proj++;
	CHECK_LE(p->_epsc->cpu_vector()->size(), p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	CHECK_LE(p->_bj->mutable_cpu_vector()->size(), p->_dim_proj * p->_dim_hcu * p->_dim_mcu);
	_epsc = p->_epsc;
	_bj = p->_bj;

	float dt;
	CHECK(_glv.getf("dt", dt));

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
	if(proj_param.has_tauepsc()){
		_tauepscdt = dt/proj_param.tauepsc();
	}else{
		_tauepscdt = _tauzidt;
	}

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
	float kepsc
){
	int offset_wij = (index/dim_mcu)*dim_conn*dim_mcu+index%dim_mcu;
	int offset_siq = (index/dim_mcu)*dim_conn;
	const float *ptr_wij_0 = ptr_wij+offset_wij;
	const int8_t *ptr_siq_0 = ptr_siq+offset_siq;
	float epsc = ptr_epsc[index] * (1-kepsc);
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
			_tauepscdt
		);
	}
}

void Proj::update_ij_cpu(){
	float prn;
	CHECK(_glv.getf("prn", prn));
	
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
	float prn;
	CHECK(_glv.getf("prn", prn));
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
	float prn;
	CHECK(_glv.getf("prn", prn));
	
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
}
}
