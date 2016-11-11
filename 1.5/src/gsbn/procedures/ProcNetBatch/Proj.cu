#include "gsbn/procedures/ProcNetGroup/Proj.hpp"

#ifndef CPU_ONLY

namespace gsbn{
namespace proc_net_group{

__global__ void update_j_kernel_gpu(
	int n,
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
	CUDA_KERNEL_LOOP(idx, n){
		float pj = ptr_pj[idx];
		float ej = ptr_ej[idx];
		float zj = ptr_zj[idx];
		int sj = ptr_sj[idx];
		
		ptr_epsc[idx] *= (1-kzi);

		if(kp){
			float bj = bgain * log(pj + eps);
			ptr_bj[idx]=bj;
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
}

void Proj::update_gpu(){
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf[Database::IDX_CONF_PRN];
	float *ptr_pj = _pj->mutable_gpu_data()+_offset_in_pop;
	float *ptr_ej = _ej->mutable_gpu_data()+_offset_in_pop;
	float *ptr_zj = _zj->mutable_gpu_data()+_offset_in_pop;
	float *ptr_bj = _bj->mutable_gpu_data()+_offset_in_pop;
	float *ptr_epsc = _epsc->mutable_gpu_data()+_offset_in_pop;
	const int *ptr_sj = _sj->gpu_data()+_offset_in_spk;

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
		_eps
	);
	CUDA_POST_KERNEL_CHECK;
}

}
}

#endif
