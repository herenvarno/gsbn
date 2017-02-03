#include "gsbn/procedures/ProcUpdPeriodic/Proj.hpp"

#ifndef CPU_ONLY

namespace gsbn{
namespace proc_upd_periodic{

__global__ void update_ssi_kernel_gpu(
	int n,
	const int *ptr_ii,
	const int *ptr_di,
	const int8_t *ptr_si,
	int *ptr_qi,
	int8_t *ptr_ssi
){
	CUDA_KERNEL_LOOP(i, n){
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
}

__global__ void update_zep_kernel_gpu(
	int dim_conn,
	int dim_mcu,
	const int8_t *ptr_ssi,
	const int8_t *ptr_sj,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	float *ptr_pj,
	float *ptr_ej,
	float *ptr_zj,
	float *ptr_epsc,
	float *ptr_bj,
	float *ptr_pij,
	float *ptr_eij,
	float *ptr_wij,
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
	
	extern __shared__ float shmem0[];
	
	int hcu_idx = blockIdx.x;
	int mcu_idx = threadIdx.x;
	
	int offset_i = hcu_idx*dim_conn;
	int offset_j = hcu_idx*dim_mcu;
	int offset_ij = hcu_idx*dim_conn*dim_mcu;
	
	float *ptr_pi_0 = ptr_pi+offset_i;
	float *ptr_pj_0 = ptr_pj+offset_j;
	float *ptr_pij_0 = ptr_pij+offset_ij;
	float *ptr_ei_0 = ptr_ei+offset_i;
	float *ptr_ej_0 = ptr_ej+offset_j;
	float *ptr_eij_0 = ptr_eij+offset_ij;
	float *ptr_zi_0 = ptr_zi+offset_i;
	float *ptr_zj_0 = ptr_zj+offset_j;
	float *ptr_bj_0 = ptr_bj+offset_j;
	float *ptr_epsc_0 = ptr_epsc+offset_j;
	float *ptr_wij_0 = ptr_wij+offset_ij;
	const int8_t *ptr_ssi_0 = ptr_ssi+offset_i;
	const int8_t *ptr_sj_0 = ptr_sj+offset_j;
	
	int8_t sj = ptr_sj_0[mcu_idx];
	float pj = ptr_pj_0[mcu_idx];
	float ej = ptr_ej_0[mcu_idx];
	float zj = ptr_zj_0[mcu_idx];
	float epsc = ptr_epsc_0[mcu_idx] * (1-kzi);
	
	float *sh_ptr_pi = &shmem0[0];
	float *sh_ptr_zi = &shmem0[dim_mcu];
	int *sh_ptr_ssi = (int *)(&shmem0[dim_mcu << 1]);
	int r=0;
	while(r<dim_conn){
		int index_i = r+mcu_idx;
		if(index_i>=dim_conn){
			break;
		}
		
		sh_ptr_pi[mcu_idx] = ptr_pi_0[index_i];
		sh_ptr_zi[mcu_idx] = ptr_zi_0[index_i];
		sh_ptr_ssi[mcu_idx] = ptr_ssi_0[index_i];
		__syncthreads();
		
		for(int s=0; s<dim_mcu; s++){
			int index_ij = (r+s)*dim_mcu+mcu_idx;
			float pij = ptr_pij_0[index_ij];
			float eij = ptr_eij_0[index_ij];
			
			float pi = sh_ptr_pi[s];
			float zi = sh_ptr_zi[s];
			int si = sh_ptr_ssi[s];
			
			// Wij
			float wij;
			if(kp){
				wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
				ptr_wij_0[index_ij] = wij;
			}else{
				wij = ptr_wij_0[index_ij];
			}
			if(si>0){
				epsc += wij;
			}
			
			// ij
			pij += (eij - pij)*kp;
			eij += (zi*zj - eij)*ke;
			ptr_pij_0[index_ij] = pij;
			ptr_eij_0[index_ij] = eij;
		}
		
		__syncthreads();
		
		float pi = sh_ptr_pi[mcu_idx];
		float zi = sh_ptr_zi[mcu_idx];
		float ei = ptr_ei_0[index_i];
		int si = sh_ptr_ssi[mcu_idx];
		
		// update i
		pi += (ei - pi)*kp;
		ei += (zi - ei)*ke;
		zi *= (1-kzi);
		if(si>0){
			zi += kfti;
		}
		ptr_pi_0[index_i] = pi;
		ptr_ei_0[index_i] = ei;
		ptr_zi_0[index_i] = zi;
		
		r += dim_mcu;
	}
	
	// update j
	if(kp){
		float bj = bgain * log(pj + eps);
		ptr_bj_0[mcu_idx] = bj;
	}
	pj += (ej - pj)*kp;
	ej += (zj - ej)*ke;
	zj *= (1-kzj);
	if(sj>0){
		zj += kftj;
	}
	ptr_pj_0[mcu_idx] = pj;
	ptr_ej_0[mcu_idx] = ej;
	ptr_zj_0[mcu_idx] = zj;
	ptr_epsc_0[mcu_idx] = epsc;
	
	
	
}

void Proj::update_ssi_gpu(){
	const int *ptr_ii = _ii->gpu_data();
	const int *ptr_di = _di->gpu_data();
	const int8_t *ptr_si = _si->gpu_data();
	int *ptr_qi = _qi->mutable_gpu_data();
	int8_t *ptr_ssi = _ssi->mutable_gpu_data();
	
	update_ssi_kernel_gpu<<<GSBN_GET_BLOCKS(_dim_hcu* _dim_conn), GSBN_GET_THREADS(_dim_hcu* _dim_conn), 0, _stream>>>(
		_dim_hcu * _dim_conn,
		ptr_ii,
		ptr_di,
		ptr_si,
		ptr_qi,
		ptr_ssi
	);
	CUDA_POST_KERNEL_CHECK;
}


void Proj::update_zep_gpu(){
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf[Database::IDX_CONF_PRN];
	float *ptr_pi = _pi->mutable_gpu_data();
	float *ptr_ei = _ei->mutable_gpu_data();
	float *ptr_zi = _zi->mutable_gpu_data();
	float *ptr_pj = _pj->mutable_gpu_data();
	float *ptr_ej = _ej->mutable_gpu_data();
	float *ptr_zj = _zj->mutable_gpu_data();
	float *ptr_bj = _bj->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_epsc = _epsc->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_pij = _pij->mutable_gpu_data();
	float *ptr_eij = _eij->mutable_gpu_data();
	float *ptr_wij = _wij->mutable_gpu_data();
	const int8_t *ptr_ssi = _ssi->gpu_data();
	const int8_t *ptr_sj = _sj->gpu_data();
	
	CUDA_CHECK(cudaFuncSetCacheConfig(update_zep_kernel_gpu, cudaFuncCachePreferShared));
	update_zep_kernel_gpu<<<_dim_hcu, _dim_mcu, 3*_dim_mcu*sizeof(float), _stream>>>(
		_dim_conn,
		_dim_mcu,
		ptr_ssi,
		ptr_sj,
		ptr_pi,
		ptr_ei,
		ptr_zi,
		ptr_pj,
		ptr_ej,
		ptr_zj,
		ptr_epsc,
		ptr_bj,
		ptr_pij,
		ptr_eij,
		ptr_wij,
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
	CUDA_POST_KERNEL_CHECK;
}

}
}

#endif
