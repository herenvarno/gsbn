#include "gsbn/procedures/ProcNet/Conn.hpp"

#ifndef CPU_ONLY

namespace gsbn{
namespace proc_net{


void Conn::add_row_gpu(int src_mcu, int delay){
        _h++;
        _ii->mutable_cpu_vector()->push_back(src_mcu);
        _qi->mutable_cpu_vector()->push_back(0);
        _di->mutable_cpu_vector()->push_back(delay);
        _si->mutable_cpu_vector()->push_back(0);
        _pi->mutable_gpu_vector()->push_back(_pi0);
        _ei->mutable_gpu_vector()->push_back(0);
        _zi->mutable_gpu_vector()->push_back(0);
        _ti->mutable_gpu_vector()->push_back(0);
        _pij->mutable_gpu_vector()->resize(_h*_w, _pi0/_w);
        _eij->mutable_gpu_vector()->resize(_h*_w, 0.0);
        _zi2->mutable_gpu_vector()->resize(_h*_w, 0.0);
        _zj2->mutable_gpu_vector()->resize(_h*_w, 0.0);
        _tij->mutable_gpu_vector()->resize(_h*_w, 0);
        _wij->mutable_gpu_vector()->resize(_h*_w, 0.0);

}


__global__ void full_update_i_kernel_gpu(
	const int n,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	const int simstep,
	const float kp,
	const float ke,
	const float kzi
){
	CUDA_KERNEL_LOOP(idx, n) {
		int index = idx;
		float zi = ptr_zi[index];
		int ti = ptr_ti[index];
		int pdt = simstep - ti;
		if(pdt<=0){
			ptr_ti[index]=simstep;
			return;
		}
	
		float pi = ptr_pi[index];
		float ei = ptr_ei[index];
	
		pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
			(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
			((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
			(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
		ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
			(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
		zi = zi*exp(-kzi*pdt);
		ti = simstep;
		
		ptr_pi[index] = pi;
		ptr_ei[index] = ei;
		ptr_zi[index] = zi;
		ptr_ti[index] = ti;
	}
}

__global__ void full_update_ij_kernel_gpu(
	const int n,
	const int w,
	const float *ptr_pi,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	float *ptr_wij,
	const int simstep,
	const float kp,
	const float ke,
	const float kzi,
	const float kzj,
	const float wgain,
	const float eps,
	const float eps2
){
	CUDA_KERNEL_LOOP(idx, n) {
		int row = idx/w;
		int col = idx%w;
		int index = idx;
	
		float pij = ptr_pij[index];
		int tij = ptr_tij[index];
		float zi = ptr_zi2[index];
		int pdt = simstep - tij;
		if(pdt<=0){
			ptr_tij[index]=simstep;
		}else{
			float eij = ptr_eij[index];
			float zj = ptr_zj2[index];
	
			pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj)/(ke - kp) -
				(ke*kp*zi*zj)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
				((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj))/(ke - kp) -
				(ke*kp*zi*zj*exp(kp*pdt - kzi*pdt - kzj*pdt))/
				(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
			eij = (eij + (ke*zi*zj)/(kzi - ke + kzj))/exp(ke*pdt) -
				(ke*zi*zj)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
			zi = zi*exp(-kzi*pdt);
			zj = zj*exp(-kzj*pdt);
			tij = simstep;
				 	
			ptr_pij[index] = pij;
			ptr_eij[index] = eij;
			ptr_zi2[index] = zi;
			ptr_zj2[index] = zj;
			ptr_tij[index] = tij;
		}
	
		// update wij and epsc
		float wij;
		if(kp){
			float pj = ptr_pj[col];
			float pi = ptr_pi[row];
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
		}
	}
}

__global__ void update_i_kernel_gpu(
	const int n,
	const int *ptr_ssi,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	const int simstep,
	const float kp,
	const float ke,
	const float kzi,
	const float kfti
){
	CUDA_KERNEL_LOOP(idx, n) {
		int index = ptr_ssi[idx];
		float zi = ptr_zi[index];
		int ti = ptr_ti[index];
		int pdt = simstep - ti;
		if(pdt<=0){
			zi += kfti;
			ptr_ti[index]=simstep;
			return;
		}
	
		float pi = ptr_pi[index];
		float ei = ptr_ei[index];
	
	
		pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
			(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
			((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
			(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
		ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
			(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
		zi = zi*exp(-kzi*pdt) + kfti;
		ti = simstep;
		ptr_pi[index] = pi;
		ptr_ei[index] = ei;
		ptr_zi[index] = zi;
		ptr_ti[index] = ti;
	}

}

__global__ void update_j_kernel_gpu(
	const int n,
	const int *ptr_sj,
	float *ptr_pj,
	float *ptr_ej,
	float *ptr_zj,
	float *ptr_bj,
	float *ptr_epsc,
	const float kp,
	const float ke,
	const float kzj,
	const float kzi,
	const float kftj,
	const float bgain,
	const float eps
){
	CUDA_KERNEL_LOOP(idx, n) {
		int index = idx;	
	
		float pj = ptr_pj[index];
		float ej = ptr_ej[index];
		float zj = ptr_zj[index];
		float sj = ptr_sj[index];
	
		// update epsc: decrease with kzi
		ptr_epsc[index] *= (1-kzi);
	
		// update bj
		if(kp){
			float bj = bgain * log(pj + eps);
			ptr_bj[index]=bj;
		}
	
		// update ZEP
		pj += (ej - pj)*kp;
		ej += (zj - ej)*ke;
		if(sj>0){
			zj *= (1-kzj) + kftj;
		}else{
			zj *= (1-kzj);
		}
		
		ptr_pj[index] = pj;
		ptr_ej[index] = ej;
		ptr_zj[index] = zj;
	}
}

__global__ void update_ij_row_kernel_gpu(
	const int n,
	const int w,
	const int *ptr_ssi,
	const float *ptr_pi,
	const float *ptr_pj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	float *ptr_wij,
	float *ptr_epsc,
	const int simstep,
	const float kp,
	const float ke,
	const float kzi,
	const float kzj,
	const float kfti,
	const float wgain,
	const float eps,
	const float eps2
){
	CUDA_KERNEL_LOOP(idx, n) {
		int row = ptr_ssi[idx/w];
		int col = idx%w;
		int index = row*w+col;
	
		float pij = ptr_pij[index];
		int tij = ptr_tij[index];
		float zi = ptr_zi2[index];
		int pdt = simstep - tij;
		if(pdt<=0){
			zi += kfti;
			ptr_zi2[index]=zi;
			ptr_tij[index]=simstep;
		
		}else{
			float eij = ptr_eij[index];
			float zj = ptr_zj2[index];
	
			pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj)/(ke - kp) -
				(ke*kp*zi*zj)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
				((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj))/(ke - kp) -
				(ke*kp*zi*zj*exp(kp*pdt - kzi*pdt - kzj*pdt))/
				(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
			eij = (eij + (ke*zi*zj)/(kzi - ke + kzj))/exp(ke*pdt) -
				(ke*zi*zj)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
			zi = zi*exp(-kzi*pdt)+kfti;
			zj = zj*exp(-kzj*pdt);
			tij = simstep;
				 	
			ptr_pij[index] = pij;
			ptr_eij[index] = eij;
			ptr_zi2[index] = zi;
			ptr_zj2[index] = zj;
			ptr_tij[index] = tij;
		}
	
		// update wij and epsc
		float wij;
		if(kp){
			float pj = ptr_pj[col];
			float pi = ptr_pi[row];
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
		}else{
			wij = ptr_wij[index];
		}
	
		// update epsc
		ptr_epsc[col] += wij;
	}
}

__global__ void update_ij_col_kernel_gpu(
	const int n,
	const int h,
	const int w,
	const int *ptr_ssj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_zi2,
	float *ptr_zj2,
	int *ptr_tij,
	const int simstep,
	const float kp,
	const float ke,
	const float kzi,
	const float kzj,
	const float kftj
){
	CUDA_KERNEL_LOOP(idx, n) {
		int row = idx%h;
		int col = ptr_ssj[idx/h];
		int index = row*w+col;
	
		int tij = ptr_tij[index];
		float zj = ptr_zj2[index];
		int pdt = simstep - tij;
		if(pdt<=0){
			zj += kftj;
			ptr_zj2[index]=zj;
			ptr_tij[index]=simstep;
			return;
		}
	
		float pij = ptr_pij[index];
		float eij = ptr_eij[index];
		float zi = ptr_zi2[index];
	
		pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj)/(ke - kp) -
			(ke*kp*zi*zj)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
			((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj))/(ke - kp) -
			(ke*kp*zi*zj*exp(kp*pdt - kzi*pdt - kzj*pdt))/
			(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
		eij = (eij + (ke*zi*zj)/(kzi - ke + kzj))/exp(ke*pdt) -
			(ke*zi*zj)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
		zi = zi*exp(-kzi*pdt);
		zj = zj*exp(-kzj*pdt)+kftj;
		tij = simstep;
			 	
		ptr_pij[index] = pij;
		ptr_eij[index] = eij;
		ptr_zi2[index] = zi;
		ptr_zj2[index] = zj;
		ptr_tij[index] = tij;
		
		// update wij
		// no need to update wij, since wij only useful when row is active.
	}
}

void Conn::update_gpu_1(){
        const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
        const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
        int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
        float prn = ptr_conf1[Database::IDX_CONF_PRN];
        float old_prn = ptr_conf1[Database::IDX_CONF_OLD_PRN];

        if(_h>0){
        if(old_prn!=prn){
                        // row update : update i (ZEPi)
                float *ptr_pi = _pi->mutable_gpu_data();
                float *ptr_ei = _ei->mutable_gpu_data();
                float *ptr_zi = _zi->mutable_gpu_data();
                int *ptr_ti = _ti->mutable_gpu_data();

                full_update_i_kernel_gpu<<<GSBN_GET_BLOCKS(_h), GSBN_GET_THREADS(_h), 0, _stream>>>(
                        _h,
                        ptr_pi,
                        ptr_ei,
                        ptr_zi,
                        ptr_ti,
                        simstep,
                        _taupdt*old_prn,
                        _tauedt,
                        _tauzidt
                );
                CUDA_POST_KERNEL_CHECK;
	}
	}
}
void Conn::update_gpu_2(){
        const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
        const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
        int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
        float prn = ptr_conf1[Database::IDX_CONF_PRN];
        float old_prn = ptr_conf1[Database::IDX_CONF_OLD_PRN];

        if(_h>0){
        if(old_prn!=prn){

                // row update: update ij (ZEPij, wij, epsc)
                float *ptr_pi = _pi->mutable_gpu_data();
                float *ptr_pj = _pj->mutable_gpu_data();
                float *ptr_pij = _pij->mutable_gpu_data();
                float *ptr_eij = _eij->mutable_gpu_data();
                float *ptr_zi2 = _zi2->mutable_gpu_data();
                float *ptr_zj2 = _zj2->mutable_gpu_data();
                int *ptr_tij = _tij->mutable_gpu_data();
                float *ptr_wij = _wij->mutable_gpu_data();

                full_update_ij_kernel_gpu<<<GSBN_GET_BLOCKS(_h*_w), GSBN_GET_THREADS(_h*_w), 0, _stream>>>(
                        _h*_w,
                        _w,
                        ptr_pi,
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
		CUDA_POST_KERNEL_CHECK;
	}
	}
}
void Conn::update_gpu_3(){
      // get active in spike
	CONST_HOST_VECTOR(int, *v_spike) = _spike->cpu_vector();
        CONST_HOST_VECTOR(int, *v_ii) = _ii->cpu_vector();
        CONST_HOST_VECTOR(int, *v_di) = _di->cpu_vector();
        HOST_VECTOR(int, *v_qi) = _qi->mutable_cpu_vector();
        HOST_VECTOR(int, *v_si) = _si->mutable_cpu_vector();
        HOST_VECTOR(int, *v_ssi) = _ssi->mutable_cpu_vector();

        if(_h>0){
        v_ssi->clear();
        for(int i=0; i<_h; i++){

                (*v_qi)[i] >>= 1;
                if((*v_qi)[i] & 0x01){
                        (*v_si)[i] = 1;
                        v_ssi->push_back(i);
                }else{
                        (*v_si)[i] = 0;
                }

                int spk = (*v_spike)[(*v_ii)[i]];
                if(spk){
                        (*v_qi)[i] |= (0x01 << (*v_di)[i]);
                }
        }

        }
        // get active out spike
        HOST_VECTOR(int, *v_sj) = _sj->mutable_cpu_vector();
        HOST_VECTOR(int, *v_ssj) = _ssj->mutable_cpu_vector();
        v_ssj->clear();
        for(int i=0; i<_w; i++){
                if((*v_spike)[i+_mcu_start]){
                        (*v_sj)[i] = 1;
                        v_ssj->push_back(i);
                }else{
                        (*v_sj)[i] = 0;
                }
        }

}
void Conn::update_gpu_4(){
        const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
        const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
        int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
        float prn = ptr_conf1[Database::IDX_CONF_PRN];

      
        CONST_DEVICE_VECTOR(int, *v_ssi) = _ssi->gpu_vector();
        int active_row_num = v_ssi->size();
        const int *ptr_ssi = _ssi->gpu_data();

        // row update : update i (ZEPi)
        float *ptr_pi = _pi->mutable_gpu_data();
        float *ptr_ei = _ei->mutable_gpu_data();
        float *ptr_zi = _zi->mutable_gpu_data();
        int *ptr_ti = _ti->mutable_gpu_data();

        if(active_row_num>0){
        update_i_kernel_gpu<<<GSBN_GET_BLOCKS(active_row_num), GSBN_GET_THREADS(active_row_num), 0, _stream>>>(
                active_row_num,
                ptr_ssi,
                ptr_pi,
                ptr_ei,
                ptr_zi,
                ptr_ti,
                simstep,
                _taupdt*prn,
                _tauedt,
                _tauzidt,
                _kfti
        );
        CUDA_POST_KERNEL_CHECK;
        }

}
void Conn::update_gpu_5(){
        const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
        const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
        int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
        float prn = ptr_conf1[Database::IDX_CONF_PRN];

        // full update : update j (ZEPj, bj)
        float *ptr_pj = _pj->mutable_gpu_data();
        float *ptr_ej = _ej->mutable_gpu_data();
        float *ptr_zj = _zj->mutable_gpu_data();
        float *ptr_bj = _bj->mutable_gpu_data()+_proj_start;
        float *ptr_epsc = _epsc->mutable_gpu_data()+_proj_start;
        const int *ptr_sj = _sj->gpu_data();
        update_j_kernel_gpu<<<GSBN_GET_BLOCKS(_w), GSBN_GET_THREADS(_w), 0, _stream>>>(
                _w,
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
        CUDA_POST_KERNEL_CHECK;

}
void Conn::update_gpu_6(){
        const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
        const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
        int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
        float prn = ptr_conf1[Database::IDX_CONF_PRN];

	CONST_DEVICE_VECTOR(int, *v_ssi) = _ssi->gpu_vector();
        int active_row_num = v_ssi->size();
        const int *ptr_ssi = _ssi->gpu_data();

        float *ptr_pi = _pi->mutable_gpu_data();
        float *ptr_pj = _pj->mutable_gpu_data();
	 // row update: update ij (ZEPij, wij, epsc)
        float *ptr_pij = _pij->mutable_gpu_data();
        float *ptr_eij = _eij->mutable_gpu_data();
        float *ptr_zi2 = _zi2->mutable_gpu_data();
        float *ptr_zj2 = _zj2->mutable_gpu_data();
        int *ptr_tij = _tij->mutable_gpu_data();
        float *ptr_wij = _wij->mutable_gpu_data();
        float *ptr_epsc = _epsc->mutable_gpu_data();
        
	if(active_row_num>0){
        update_ij_row_kernel_gpu<<<GSBN_GET_BLOCKS(active_row_num*_w), GSBN_GET_THREADS(active_row_num*_w), 0, _stream>>>(
                active_row_num*_w,
                _w,
                ptr_ssi,
                ptr_pi,
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
        CUDA_POST_KERNEL_CHECK;
        }

}
void Conn::update_gpu_7(){
        const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
        const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
        int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
        float prn = ptr_conf1[Database::IDX_CONF_PRN];
	
	CONST_DEVICE_VECTOR(int, *v_ssj) = _ssj->gpu_vector();
        int active_col_num = v_ssj->size();
        const int *ptr_ssj = _ssj->gpu_data();

        // col update: update ij (ZEPij)
        float *ptr_pij = _pij->mutable_gpu_data();
        float *ptr_eij = _eij->mutable_gpu_data();
        float *ptr_zi2 = _zi2->mutable_gpu_data();
        float *ptr_zj2 = _zj2->mutable_gpu_data();
        int *ptr_tij = _tij->mutable_gpu_data();

        if(active_col_num*_h>0){
        update_ij_col_kernel_gpu<<<GSBN_GET_BLOCKS(active_col_num*_h), GSBN_GET_THREADS(active_col_num*_h), 0, _stream>>>(
                active_col_num*_h,
                _h,
                _w,
                ptr_ssj,
                ptr_pij,
                ptr_eij,
                ptr_zi2,
                ptr_zj2,
                ptr_tij,
                simstep,
                _taupdt*prn,
                _tauedt,
                _tauzidt,
                _tauzjdt,
                _kftj
        );
        CUDA_POST_KERNEL_CHECK;
        }

}
void Conn::update_gpu(){
	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	float old_prn = ptr_conf1[Database::IDX_CONF_OLD_PRN];
	
	if(_h>0){
	if(old_prn!=prn){
			// row update : update i (ZEPi)
		float *ptr_pi = _pi->mutable_gpu_data();
		float *ptr_ei = _ei->mutable_gpu_data();
		float *ptr_zi = _zi->mutable_gpu_data();
		int *ptr_ti = _ti->mutable_gpu_data();

		full_update_i_kernel_gpu<<<GSBN_GET_BLOCKS(_h), GSBN_GET_THREADS(_h), 0, _stream>>>(
			_h,
			ptr_pi,
			ptr_ei,
			ptr_zi,
			ptr_ti,
			simstep,
			_taupdt*old_prn,
			_tauedt,
			_tauzidt
		);
		CUDA_POST_KERNEL_CHECK;

		// row update: update ij (ZEPij, wij, epsc)
		float *ptr_pj = _pj->mutable_gpu_data();
		float *ptr_pij = _pij->mutable_gpu_data();
		float *ptr_eij = _eij->mutable_gpu_data();
		float *ptr_zi2 = _zi2->mutable_gpu_data();
		float *ptr_zj2 = _zj2->mutable_gpu_data();
		int *ptr_tij = _tij->mutable_gpu_data();
		float *ptr_wij = _wij->mutable_gpu_data();
	
		full_update_ij_kernel_gpu<<<GSBN_GET_BLOCKS(_h*_w), GSBN_GET_THREADS(_h*_w), 0, _stream>>>(
			_h*_w,
			_w,
			ptr_pi,
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
		CUDA_POST_KERNEL_CHECK;
	}
	}
	// get active in spike
	CONST_DEVICE_VECTOR(int, *v_spike) = _spike->gpu_vector();
	CONST_DEVICE_VECTOR(int, *v_ii) = _ii->gpu_vector();
	CONST_DEVICE_VECTOR(int, *v_di) = _di->gpu_vector();
	DEVICE_VECTOR(int, *v_qi) = _qi->mutable_gpu_vector();
	DEVICE_VECTOR(int, *v_si) = _si->mutable_gpu_vector();
	DEVICE_VECTOR(int, *v_ssi) = _ssi->mutable_gpu_vector();
	
	if(_h>0){
	v_ssi->clear();
	for(int i=0; i<_h; i++){
		
		(*v_qi)[i] >>= 1;
		if((*v_qi)[i] & 0x01){
			(*v_si)[i] = 1;
			v_ssi->push_back(i);
		}else{
			(*v_si)[i] = 0;
		}
	
		int spk = (*v_spike)[(*v_ii)[i]];
		if(spk){
			(*v_qi)[i] |= (0x01 << (*v_di)[i]);
		}
	}

	}
	// get active out spike
	DEVICE_VECTOR(int, *v_sj) = _sj->mutable_gpu_vector();
	DEVICE_VECTOR(int, *v_ssj) = _ssj->mutable_gpu_vector();
	v_ssj->clear();
	for(int i=0; i<_w; i++){
		if((*v_spike)[i+_mcu_start]){
			(*v_sj)[i] = 1;
			v_ssj->push_back(i);
		}else{
			(*v_sj)[i] = 0;
		}
	}
	
	int active_row_num = v_ssi->size();
	int active_col_num = v_ssj->size();
	const int *ptr_ssi = _ssi->gpu_data();
	const int *ptr_ssj = _ssj->gpu_data();
	
	// row update : update i (ZEPi)
	float *ptr_pi = _pi->mutable_gpu_data();
	float *ptr_ei = _ei->mutable_gpu_data();
	float *ptr_zi = _zi->mutable_gpu_data();
	int *ptr_ti = _ti->mutable_gpu_data();
	
	if(active_row_num>0){
	update_i_kernel_gpu<<<GSBN_GET_BLOCKS(active_row_num), GSBN_GET_THREADS(active_row_num), 0, _stream>>>(
		active_row_num,
		ptr_ssi,
		ptr_pi,
		ptr_ei,
		ptr_zi,
		ptr_ti,
		simstep,
		_taupdt*prn,
		_tauedt,
		_tauzidt,
		_kfti
	);
	CUDA_POST_KERNEL_CHECK;
	}

	// full update : update j (ZEPj, bj)
	float *ptr_pj = _pj->mutable_gpu_data();
	float *ptr_ej = _ej->mutable_gpu_data();
	float *ptr_zj = _zj->mutable_gpu_data();
	float *ptr_bj = _bj->mutable_gpu_data()+_proj_start;
	float *ptr_epsc = _epsc->mutable_gpu_data()+_proj_start;
	const int *ptr_sj = _sj->gpu_data();
	update_j_kernel_gpu<<<GSBN_GET_BLOCKS(_w), GSBN_GET_THREADS(_w), 0, _stream>>>(
		_w,
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
	CUDA_POST_KERNEL_CHECK;
	
	// row update: update ij (ZEPij, wij, epsc)
	float *ptr_pij = _pij->mutable_gpu_data();
	float *ptr_eij = _eij->mutable_gpu_data();
	float *ptr_zi2 = _zi2->mutable_gpu_data();
	float *ptr_zj2 = _zj2->mutable_gpu_data();
	int *ptr_tij = _tij->mutable_gpu_data();
	float *ptr_wij = _wij->mutable_gpu_data();
	
	if(active_row_num>0){
	update_ij_row_kernel_gpu<<<GSBN_GET_BLOCKS(active_row_num*_w), GSBN_GET_THREADS(active_row_num*_w), 0, _stream>>>(
		active_row_num*_w,
		_w,
		ptr_ssi,
		ptr_pi,
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
	CUDA_POST_KERNEL_CHECK;
	}
	// col update: update ij (ZEPij)
	if(active_col_num*_h>0){
	update_ij_col_kernel_gpu<<<GSBN_GET_BLOCKS(active_col_num*_h), GSBN_GET_THREADS(active_col_num*_h), 0, _stream>>>(
		active_col_num*_h,
		_h,
		_w,
		ptr_ssj,
		ptr_pij,
		ptr_eij,
		ptr_zi2,
		ptr_zj2,
		ptr_tij,
		simstep,
		_taupdt*prn,
		_tauedt,
		_tauzidt,
		_tauzjdt,
		_kftj
	);
	CUDA_POST_KERNEL_CHECK;
	}
}

}
}

#endif
