#include "gsbn/procedures/ProcNetBatch/Pop.hpp"

#ifndef CPU_ONLY

namspace gsbn{
namespace proc_net_batch{

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
	int* ptr_spk,
	float wgain,
	float lgbias,
	float igain,
	float taumdt,
	float wtagain,
	float maxfqdt
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
	ptr_spk[idx] = int(ptr_rnd_uniform01[idx]<act*maxfqdt);

}

void Pop::update_sup_gpu(){
	const int *ptr_conf = static_cast<const int*>(_conf->cpu_data());
	int lginp_idx= ptr_conf[Database::IDX_CONF_STIM];
	int wmask_idx= ptr_conf[Database::IDX_CONF_GAIN_MASK]+_hcu_start;
	const float* ptr_wmask = _wmask->gpu_data(wmask_idx);
	const float* ptr_epsc = _epsc->gpu_data();
	const float* ptr_bj = _bj->gpu_data();
	const float *ptr_lginp = _lginp->gpu_data(lginp_idx)+_mcu_start;
	const float *ptr_rnd_normal = _rnd_normal->gpu_data();
	const float *ptr_rnd_uniform01 = _rnd_uniform01->gpu_data();
	float *ptr_dsup = _dsup->mutable_gpu_data();
	float *ptr_act = _act->mutable_gpu_data();
	int *ptr_spk = _spike->mutable_gpu_data()+_mcu_start;
	
	update_kernel_gpu<<<dim_hcu, dim_mcu, dim_mcu*sizeof(float), _stream>>>(
		dim_proj,
		dim_hcu,
		dim_mcu,
		ptr_epsc,
		ptr_bj,
		ptr_lginp,
		ptr_wmask,
		ptr_rnd_normal,
		ptr_rnd_uniform01,
		ptr_dsup,
		ptr_act,
		ptr_spk,
		_wgain,
		_lgbias,
		_igain,
		_taumdt,
		_wtagain,
		_maxfqdt
	);
	CUDA_POST_KERNEL_CHECK;
}


}
}

#endif
