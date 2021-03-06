#include "gsbn/procedures/ProcUpdLazy/Pop.hpp"

#ifndef CPU_ONLY

namespace gsbn{
namespace proc_upd_lazy{

void Pop::update_rnd_gpu(){
	float *ptr_uniform01= _rnd_uniform01->mutable_gpu_data();
	float *ptr_normal= _rnd_normal->mutable_gpu_data();
	int size = _dim_hcu * _dim_mcu;
	_rnd.gen_uniform01_gpu(ptr_uniform01, size);
	_rnd.gen_normal_gpu(ptr_normal, size, 0, _snoise);
}


__global__ void update_sup_kernel_gpu(
	int dim_proj,
	int dim_hcu,
	int dim_mcu,
	const float *ptr_epsc,
	const float *ptr_bj,
	const float *ptr_lginp,
	const float *ptr_wmask,
	const float *ptr_rnd_normal,
	const float *ptr_rnd_uniform01,
	float* ptr_dsup,
	float* ptr_act,
	int8_t* ptr_spk,
	float* ptr_ada,
	int* ptr_counter,
	float wgain,
	float lgbias,
	float igain,
	float taumdt,
	float wtagain,
	float maxfqdt,
	float adgain,
	float ka
){
	extern __shared__ float shmem[];

	int i=blockIdx.x;
	int j=threadIdx.x;
	int idx = i*dim_mcu+j;

	float wsup=0;
	int offset=0;
	int mcu_num_in_pop = dim_hcu * dim_mcu;
	for(int m=0; m<dim_proj; m++){
		wsup += ptr_bj[offset+idx] + ptr_epsc[offset+idx];
		offset += mcu_num_in_pop;
	}

	__shared__ float wmask;
	if(j==0){
		wmask = ptr_wmask[i];
	}
	
	__syncthreads();
	float sup = lgbias + igain * ptr_lginp[idx]+ ptr_rnd_normal[idx];
	sup += (wgain * wmask) * wsup;
//	float ada = ptr_ada[idx];
//	sup -= ada;

	float dsup = ptr_dsup[idx];
	dsup += (sup - dsup)*taumdt;
	ptr_dsup[idx] = dsup;
	
	float* ptr_sh_dsup=&shmem[0];
	ptr_sh_dsup[j] = dsup;
	__syncthreads();
	if(j==0){
		for(int n=1; n<dim_mcu; n++){
			if(ptr_sh_dsup[0]<ptr_sh_dsup[n]){
				ptr_sh_dsup[0] = ptr_sh_dsup[n];
			}
		}
	}
	__syncthreads();
	float maxdsup = ptr_sh_dsup[0];
	float maxact = exp(wtagain*maxdsup);
	float act = exp(wtagain*(dsup-maxdsup));
	if(maxact<1){
		act *= maxact;
	}
	float* ptr_sh_act=&shmem[0];
	__syncthreads();
	ptr_sh_act[j] = act;
	__syncthreads();
	if(j==0){
		for(int n=1; n<dim_mcu; n++){
			ptr_sh_act[0]+=ptr_sh_act[n];
		}
	}
	__syncthreads();
	float vsum = ptr_sh_act[0];
	if(vsum>1){
		act /= vsum;
	}

	ptr_act[idx] = act;
	int8_t spk = int8_t(ptr_rnd_uniform01[idx]<act*maxfqdt);
	ptr_spk[idx] = spk;
//	ptr_counter[idx] += spk;
//	ptr_ada[idx] += (adgain * act - ada) * ka;

}

void Pop::update_sup_gpu(){
	int lginp_idx;
	int wmask_idx;
	CHECK(_glv.geti("lginp-idx", lginp_idx));
	CHECK(_glv.geti("wmask-idx", wmask_idx));
	const float *ptr_lginp = _lginp->gpu_data(lginp_idx)+_mcu_start;
	const float* ptr_wmask = _wmask->gpu_data(wmask_idx)+_hcu_start;
	const float* ptr_epsc = _epsc->gpu_data();
	const float* ptr_bj = _bj->gpu_data();
	const float *ptr_rnd_normal = _rnd_normal->gpu_data();
	const float *ptr_rnd_uniform01 = _rnd_uniform01->gpu_data();
	float *ptr_dsup = _dsup->mutable_gpu_data();
	float *ptr_act = _act->mutable_gpu_data();
	int8_t *ptr_spk = _spike->mutable_gpu_data();
	int *ptr_counter = _counter->mutable_gpu_data();
	float *ptr_ada = _ada->mutable_gpu_data();

	update_sup_kernel_gpu<<<_dim_hcu, _dim_mcu, _dim_mcu*sizeof(float), _stream>>>(
		_dim_proj,
		_dim_hcu,
		_dim_mcu,
		ptr_epsc,
		ptr_bj,
		ptr_lginp,
		ptr_wmask,
		ptr_rnd_normal,
		ptr_rnd_uniform01,
		ptr_dsup,
		ptr_act,
		ptr_spk,
		ptr_ada,
		ptr_counter,
		_wgain,
		_lgbias,
		_igain,
		_taumdt,
		_wtagain,
		_maxfqdt,
		_adgain,
		_tauadt
	);
	CUDA_POST_KERNEL_CHECK;
}


}
}

#endif
