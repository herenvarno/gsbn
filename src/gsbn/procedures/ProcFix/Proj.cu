#include "gsbn/procedures/ProcFix/Proj.hpp"

#ifndef CPU_ONLY

#include "gsbn/Fix16.cuh"

namespace gsbn{
namespace proc_fix{

__global__ void update_full_kernel_gpu(
	int dim_conn,
	int dim_mcu,
	fix16 *ptr_pi,
	fix16 *ptr_ei,
	fix16 *ptr_zi,
	int *ptr_ti,
	const fix16 *ptr_pj,
	fix16 *ptr_pij,
	fix16 *ptr_eij,
	fix16 *ptr_zi2,
	fix16 *ptr_zj2,
	int *ptr_tij,
	fix16 *ptr_wij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float wgain,
	float eps,
	float eps2,
	int norm_frac_bit,
	int p_frac_bit
){
	int i=blockIdx.y*gridDim.x+blockIdx.x;
	int j=threadIdx.x;
	
	__shared__ float sh_pi;
	if(j==0){
		float pi = fix16_to_fp32_gpu(ptr_pi[i], p_frac_bit);
		float zi = fix16_to_fp32_gpu(ptr_zi[i], norm_frac_bit);
		int ti = ptr_ti[i];
		int pdt = simstep - ti;
		if(pdt<=0){
			ptr_ti[i]=simstep;
		}else{
			float ei = fix16_to_fp32_gpu(ptr_ei[i], norm_frac_bit);
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
				(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
				((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
				(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
				(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			zi = zi*exp(-kzi*pdt);
			ti = simstep;
		
			ptr_pi[i] = fp32_to_fix16_gpu(pi, p_frac_bit);
			ptr_ei[i] = fp32_to_fix16_gpu(ei, norm_frac_bit);
			ptr_zi[i] = fp32_to_fix16_gpu(zi, norm_frac_bit);
			ptr_ti[i] = ti;
		}
		sh_pi = pi;
	}
	__syncthreads();
	
	int index = i*dim_mcu+j;
	
	int tij = ptr_tij[index];
	float zi2 = fix16_to_fp32_gpu(ptr_zi2[index], norm_frac_bit);
	int pdt = simstep - tij;
	if(pdt<=0){
		ptr_tij[index]=simstep;
	}else{
		float pij = fix16_to_fp32_gpu(ptr_pij[index], p_frac_bit);
		float eij = fix16_to_fp32_gpu(ptr_eij[index], norm_frac_bit);
		float zj2 = fix16_to_fp32_gpu(ptr_zj2[index], norm_frac_bit);
	
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
			 	
		ptr_pij[index] = fp32_to_fix16_gpu(pij, p_frac_bit);
		ptr_eij[index] = fp32_to_fix16_gpu(eij, norm_frac_bit);
		ptr_zi2[index] = fp32_to_fix16_gpu(zi2, norm_frac_bit);
		ptr_zj2[index] = fp32_to_fix16_gpu(zj2, norm_frac_bit);
		ptr_tij[index] = tij;
			
		// update wij and epsc
		float wij;
		if(kp){
			float pi = sh_pi;
			float pj = fix16_to_fp32_gpu(ptr_pj[i/dim_conn*dim_mcu + j], p_frac_bit);
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = fp32_to_fix16_gpu(wij, norm_frac_bit);
		}
	}
}

__global__ void update_j_kernel_gpu(
	int n,
	const int8_t *ptr_sj,
	fix16 *ptr_pj,
	fix16 *ptr_ej,
	fix16 *ptr_zj,
	fix16 *ptr_bj,
	fix16 *ptr_epsc,
	float kp,
	float ke,
	float kzj,
	float kzi,
	float kftj,
	float bgain,
	float eps,
	int norm_frac_bit,
	int p_frac_bit
){
	CUDA_KERNEL_LOOP(idx, n){
		float pj = fix16_to_fp32_gpu(ptr_pj[idx], p_frac_bit);
		float ej = fix16_to_fp32_gpu(ptr_ej[idx], norm_frac_bit);
		float zj = fix16_to_fp32_gpu(ptr_zj[idx], norm_frac_bit);
		int8_t sj = ptr_sj[idx];

		float epsc = fix16_to_fp32_gpu(ptr_epsc[idx], norm_frac_bit);
		ptr_epsc[idx] = fp32_to_fix16_gpu(epsc*(1-kzi), norm_frac_bit);

		if(kp){
			float bj = bgain * log(pj + eps);
			ptr_bj[idx]=fp32_to_fix16_gpu(bj, norm_frac_bit);
		}

		pj += (ej - pj)*kp;
		ej += (zj - ej)*ke;
		zj *= (1-kzj);
		if(sj>0){
			zj += kftj;
		}
	
		ptr_pj[idx] = fp32_to_fix16_gpu(pj, p_frac_bit);
		ptr_ej[idx] = fp32_to_fix16_gpu(ej, norm_frac_bit);
		ptr_zj[idx] = fp32_to_fix16_gpu(zj, norm_frac_bit);
	}
}

__global__ void update_row_kernel_gpu(
	int dim_conn,
	int dim_mcu,
	const int *ptr_ssi,
	fix16 *ptr_pi,
	fix16 *ptr_ei,
	fix16 *ptr_zi,
	int *ptr_ti,
	const fix16 *ptr_pj,
	fix16 *ptr_pij,
	fix16 *ptr_eij,
	fix16 *ptr_zi2,
	fix16 *ptr_zj2,
	int *ptr_tij,
	fix16* ptr_wij,
	fix16* ptr_epsc,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kfti,
	float wgain,
	float eps,
	float eps2,
	int norm_frac_bit,
	int p_frac_bit
){

	int i = blockIdx.x;
	int j = threadIdx.x;
	int row = ptr_ssi[i];
	int col = j;
	int index = row*dim_mcu+col;
	
	__shared__ float sh_pi;
	
	if(j==0){
		float pi = fix16_to_fp32_gpu(ptr_pi[row], p_frac_bit);
		float zi = fix16_to_fp32_gpu(ptr_zi[row], norm_frac_bit);
		int ti = ptr_ti[row];
		int pdt = simstep - ti;
		if(pdt<=0){
			zi += kfti;
			ptr_zi[row] = fp32_to_fix16_gpu(zi, norm_frac_bit);
			ptr_ti[row] = simstep;
		}else{
			float ei = fp32_to_fix16_gpu(ptr_ei[row], norm_frac_bit);
		
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
				(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
				((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
				(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
				(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			zi = zi*exp(-kzi*pdt) + kfti;
			ti = simstep;
			ptr_pi[row] = fp32_to_fix16_gpu(pi, p_frac_bit);
			ptr_ei[row] = fp32_to_fix16_gpu(ei, norm_frac_bit);
			ptr_zi[row] = fp32_to_fix16_gpu(zi, norm_frac_bit);
			ptr_ti[row] = ti;
		}
		sh_pi = pi;
	}
	
	__syncthreads();
	float pij = fix16_to_fp32_gpu(ptr_pij[index], p_frac_bit);
	int tij = ptr_tij[index];
	float zi2 = fix16_to_fp32_gpu(ptr_zi2[index], norm_frac_bit);
	int pdt = simstep - tij;
	if(pdt<=0){
		zi2 += kfti;
		ptr_zi2[index] = fp32_to_fix16_gpu(zi2, norm_frac_bit);
		ptr_tij[index] = simstep;
	}else{
		float eij = fix16_to_fp32_gpu(ptr_eij[index], norm_frac_bit);
		float zj2 = fix16_to_fp32_gpu(ptr_zj2[index], norm_frac_bit);
	
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
			 	
		ptr_pij[index] = fp32_to_fix16_gpu(pij, p_frac_bit);
		ptr_eij[index] = fp32_to_fix16_gpu(eij, norm_frac_bit);
		ptr_zi2[index] = fp32_to_fix16_gpu(zi2, norm_frac_bit);
		ptr_zj2[index] = fp32_to_fix16_gpu(zj2, norm_frac_bit);
		ptr_tij[index] = tij;
		
		float wij;
		int idx_hcu = row / dim_conn;
		int idx_mcu = idx_hcu * dim_mcu + j;
		if(kp){
			float pi = sh_pi;
			float pj = fix16_to_fp32_gpu(ptr_pj[idx_mcu], p_frac_bit);
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = fp32_to_fix16_gpu(wij, norm_frac_bit);
		}else{
			wij = fix16_to_fp32_gpu(ptr_wij[index], norm_frac_bit);
		}
		atomic_add_fp32_to_fix16_gpu(&ptr_epsc[idx_mcu], wij, norm_frac_bit);
	}
}

__global__ void update_col_kernel_gpu(
	int dim_conn,
	int dim_mcu,
	const int *ptr_ii,
	const int *ptr_ssj,
	fix16 *ptr_pij,
	fix16 *ptr_eij,
	fix16 *ptr_zi2,
	fix16 *ptr_zj2,
	int *ptr_tij,
	int simstep,
	float kp,
	float ke,
	float kzi,
	float kzj,
	float kftj,
	int norm_frac_bit,
	int p_frac_bit
){

	int i = blockIdx.x;
	int j = threadIdx.x;

	int row = ptr_ssj[j]/dim_mcu*dim_conn+i;
	if(ptr_ii[row]<0){
		return;
	}
	int col = ptr_ssj[j]%dim_mcu;
	int index = row*dim_mcu+col;
	
	int tij = ptr_tij[index];
	float zj2 = fix16_to_fp32_gpu(ptr_zj2[index], norm_frac_bit);
	int pdt = simstep - tij;
	if(pdt<=0){
		zj2 += kftj;
		ptr_zj2[index]= fp32_to_fix16_gpu(zj2, norm_frac_bit);
		ptr_tij[index]=simstep;
	}else{
		float pij = fix16_to_fp32_gpu(ptr_pij[index], p_frac_bit);
		float eij = fix16_to_fp32_gpu(ptr_eij[index], norm_frac_bit);
		float zi2 = fix16_to_fp32_gpu(ptr_zi2[index], norm_frac_bit);
	
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
			 	
		ptr_pij[index] = fp32_to_fix16_gpu(pij, p_frac_bit);
		ptr_eij[index] = fp32_to_fix16_gpu(eij, norm_frac_bit);
		ptr_zi2[index] = fp32_to_fix16_gpu(zi2, norm_frac_bit);
		ptr_zj2[index] = fp32_to_fix16_gpu(zj2, norm_frac_bit);
		ptr_tij[index] = tij;
	}
}


void Proj::update_full_gpu(){
	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	float old_prn = ptr_conf1[Database::IDX_CONF_OLD_PRN];
	if(old_prn!=prn){
		fix16 *ptr_pi = _pi->mutable_gpu_data();
		fix16 *ptr_ei = _ei->mutable_gpu_data();
		fix16 *ptr_zi = _zi->mutable_gpu_data();
		int *ptr_ti = _ti->mutable_gpu_data();
		const fix16 *ptr_pj = _pj->gpu_data();
		fix16 *ptr_pij = _pij->mutable_gpu_data();
		fix16 *ptr_eij = _eij->mutable_gpu_data();
		fix16 *ptr_zi2 = _zi2->mutable_gpu_data();
		fix16 *ptr_zj2 = _zj2->mutable_gpu_data();
		int *ptr_tij = _tij->mutable_gpu_data();
		fix16 *ptr_wij = _wij->mutable_gpu_data();
		const dim3 GRID_SIZE(_dim_conn, _dim_hcu);
		update_full_kernel_gpu<<<GRID_SIZE, _dim_mcu, 0, _stream>>>(
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
			_eps2,
			_norm_frac_bit,
			_p_frac_bit
		);
		CUDA_POST_KERNEL_CHECK;
	}
}

void Proj::update_j_gpu(){
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf[Database::IDX_CONF_PRN];
	fix16 *ptr_pj = _pj->mutable_gpu_data();
	fix16 *ptr_ej = _ej->mutable_gpu_data();
	fix16 *ptr_zj = _zj->mutable_gpu_data();
	fix16 *ptr_bj = _bj->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	fix16 *ptr_epsc = _epsc->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	const int8_t *ptr_sj = _sj->gpu_data();

	update_j_kernel_gpu<<<GSBN_GET_BLOCKS(_dim_hcu*_dim_mcu), GSBN_GET_THREADS(_dim_hcu*_dim_mcu), 0, _stream>>>(
		_dim_hcu*_dim_mcu,
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
		_eps,
		_norm_frac_bit,
		_p_frac_bit
	);
	CUDA_POST_KERNEL_CHECK;
}

void Proj::update_ss_gpu(){
	// get active in spike
	CONST_HOST_VECTOR(int8_t, *v_si) = _si->cpu_vector();
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
                if(spk){
                        (*v_qi)[i] |= (0x01 << (*v_di)[i]);
                }
        }

        // get active out spike
        CONST_HOST_VECTOR(int8_t, *v_sj) = _sj->cpu_vector();
        HOST_VECTOR(int, *v_ssj) = _ssj->mutable_cpu_vector();
        v_ssj->clear();
        for(int i=0; i<_dim_hcu * _dim_mcu; i++){
                if((*v_sj)[i]>0){
                        v_ssj->push_back(i);
                }
        }

}

void Proj::update_row_gpu(){
	int active_row_num = _ssi->gpu_vector()->size();
	if(active_row_num<=0){
		return;
	}

	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	
	fix16 *ptr_pi = _pi->mutable_gpu_data();
	fix16 *ptr_ei = _ei->mutable_gpu_data();
	fix16 *ptr_zi = _zi->mutable_gpu_data();
	int *ptr_ti = _ti->mutable_gpu_data();
	const fix16 *ptr_pj = _pj->gpu_data();
	fix16 *ptr_pij = _pij->mutable_gpu_data();
	fix16 *ptr_eij = _eij->mutable_gpu_data();
	fix16 *ptr_zi2 = _zi2->mutable_gpu_data();
	fix16 *ptr_zj2 = _zj2->mutable_gpu_data();
	int *ptr_tij = _tij->mutable_gpu_data();
	fix16 *ptr_wij = _wij->mutable_gpu_data();
	fix16 *ptr_epsc = _epsc->mutable_gpu_data()+ _proj_in_pop * _dim_hcu * _dim_mcu;
	
	const int *ptr_ssi = _ssi->gpu_data();

	update_row_kernel_gpu<<<active_row_num, _dim_mcu, 0, _stream>>>(
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
		_eps2,
		_norm_frac_bit,
		_p_frac_bit
	);
	CUDA_POST_KERNEL_CHECK;
	
	ptr_wij= _wij->mutable_cpu_data();
	for(int i=0; i<100; i++){
		cout << fix16_to_fp32(ptr_wij[i], _norm_frac_bit) << ",";
	}
	cout << endl;
}

void Proj::update_col_gpu(){
	int active_col_num = _ssj->gpu_vector()->size();
	if(active_col_num<=0){
		return;
	}
	const int *ptr_conf0 = static_cast<const int*>(_conf->cpu_data());
	const float *ptr_conf1 = static_cast<const float*>(_conf->cpu_data());
	int simstep = ptr_conf0[Database::IDX_CONF_TIMESTAMP];
	float prn = ptr_conf1[Database::IDX_CONF_PRN];
	
	fix16 *ptr_pij = _pij->mutable_gpu_data();
	fix16 *ptr_eij = _eij->mutable_gpu_data();
	fix16 *ptr_zi2 = _zi2->mutable_gpu_data();
	fix16 *ptr_zj2 = _zj2->mutable_gpu_data();
	int *ptr_tij = _tij->mutable_gpu_data();
	
	const int *ptr_ii = _ii->gpu_data();
	const int *ptr_ssj = _ssj->gpu_data();
	
	update_col_kernel_gpu<<<_dim_conn, active_col_num, 0, _stream>>>(
		_dim_conn,
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
		_kftj,
		_norm_frac_bit,
		_p_frac_bit
	);
	CUDA_POST_KERNEL_CHECK;
}

}
}

#endif
