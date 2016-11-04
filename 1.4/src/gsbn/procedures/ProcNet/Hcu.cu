#include "gsbn/procedures/ProcNet/Hcu.hpp"

#ifndef CPU_ONLY

struct sine : public thrust::unary_function<float,float>
{
  __host__ __device__
  float operator()(float x) { return sinf(x); }
};

struct _F_DIV_VSUM{
        __host__ __device__
        _F_DIV_VSUM(float vsum): _vsum(vsum){};
        float _vsum;
        __host__ __device__
        float operator()(const float act) const {
                return act / _vsum;
        }
};
struct _F_DSUP_ACT_0{
        __host__ __device__
        _F_DSUP_ACT_0(float wtagain, float maxdsup, float maxact):_wtagain(wtagain), _maxdsup(maxdsup), _maxact(maxact){};
        float _wtagain;
        float _maxdsup;
        float _maxact;
        __host__ __device__
        float operator()(const float dsup) const {
                return exp(_wtagain*(dsup-_maxdsup))*_maxact;
        }
};
struct _F_DSUP_ACT_1{
        __host__ __device__
        _F_DSUP_ACT_1(float wtagain, float maxdsup):_wtagain(wtagain), _maxdsup(maxdsup){};
        float _wtagain;
        float _maxdsup;
        __host__ __device__
        float operator()(const float dsup) const {
                return exp(_wtagain*(dsup-_maxdsup));
        }
};
struct _F_GEN_SPK{
        __host__ __device__
        _F_GEN_SPK(float maxfqdt):_maxfqdt(maxfqdt){};
        float _maxfqdt;
        __host__ __device__
        int operator()(const float act, const float rnd) const {
                return int(rnd < act * _maxfqdt);
        }
};


namespace gsbn{
namespace proc_net{

__global__ void update_dsup_kernel_gpu(
	const int n,
	const int isp_num,
	const int mcu_num,
	const float *ptr_epsc,
	const float *ptr_bj,
	const float *ptr_lginp,
	const float *ptr_rnd_normal,
	float *ptr_dsup,
	const float wgain,
	const float wmask,
	const float lgbias,
	const float igain,
	const float taumdt
){
	CUDA_KERNEL_LOOP(idx, n) {
		float wsup=0;
		int offset=0;
		for(int i=0; i<isp_num; i++){
			wsup += ptr_bj[idx+offset] + ptr_epsc[idx+offset];
			offset+=mcu_num;
		}
	
		float sup = lgbias + igain * ptr_lginp[idx] + ptr_rnd_normal[idx];
		sup += (wgain * wmask) * wsup;

		float dsup=ptr_dsup[idx];
		ptr_dsup[idx] += (sup - dsup) * taumdt;
	}
}



__global__ void update_halfnorm_kernel_gpu(
	const int n,
	const float *ptr_dsup,
	float *ptr_act,
	const int mcu_num,
	const float wtagain
){	
	CUDA_KERNEL_LOOP(idx, n) {
	float maxdsup = ptr_dsup[0];
	for(int i=0; i<mcu_num; i++){
		float dsup=ptr_dsup[i];
		if(dsup>maxdsup){
			maxdsup=dsup;
		}
	}
	float maxact = exp(wtagain*maxdsup);
	float vsum=0;
	for(int j=0; j<mcu_num; j++){
		float act = exp(wtagain*(ptr_dsup[j]-maxdsup));
		if(maxact<1){
			act *= maxact;
		}
		vsum += act;
		*(ptr_act+j)=act;
	}
	if(vsum>1){
		for(int k=0; k<mcu_num; k++){
			ptr_act[k] /= vsum;
		}
	}
	}
}

__global__ void update_spike_kernel_gpu(
	const int n,
	const float *ptr_act,
	const float *ptr_rnd_uniform01,
	int *ptr_spike,
	const float maxfqdt
){
	CUDA_KERNEL_LOOP(idx, n) {
		ptr_spike[idx] = int(ptr_rnd_uniform01[idx] < ptr_act[idx]*maxfqdt);
	}
}

void Hcu::update_gpu_1(){
        const int *ptr_conf = static_cast<const int*>(_conf->cpu_data());
        int lginp_idx= ptr_conf[Database::IDX_CONF_STIM];
        int wmask_idx= ptr_conf[Database::IDX_CONF_GAIN_MASK];
        float wmask = (_wmask->cpu_data(wmask_idx))[_id];
        const float* ptr_epsc = _epsc->gpu_data();
        const float* ptr_bj = _bj->gpu_data();
        float *ptr_dsup = _dsup->mutable_gpu_data();
        const float *ptr_lginp = _lginp->gpu_data(lginp_idx)+_mcu_start;
        const float *ptr_rnd_normal = _rnd_normal->gpu_data(_mcu_start);
        
        update_dsup_kernel_gpu<<<GSBN_GET_BLOCKS(_mcu_num), GSBN_GET_THREADS(_mcu_num), 0, _stream>>>(
                _mcu_num,
                _isp.size(),
                _mcu_num,
                ptr_epsc,
                ptr_bj,
                ptr_lginp,
                ptr_rnd_normal,
                ptr_dsup,
                _wgain,
                wmask,
                _lgbias,
                _igain,
                _taumdt
        );
        CUDA_POST_KERNEL_CHECK; 

}

void Hcu::update_gpu_2_1(){
        DEVICE_VECTOR(float, *v_dsup) = _dsup->mutable_gpu_vector();

        // halfnorm
        // thrust
        _maxdsup = thrust::reduce(thrust::cuda::par.on(_stream), v_dsup->begin(), v_dsup->end(), -1000.0f,  thrust::maximum<float>());
}
void Hcu::update_gpu_2_2(){
        DEVICE_VECTOR(float, *v_dsup) = _dsup->mutable_gpu_vector();
        DEVICE_VECTOR(float, *v_act) = _act->mutable_gpu_vector();
        float maxact = exp(_wtagain * _maxdsup);
        if(maxact<1){
                _F_DSUP_ACT_0 fdsupact0(_wtagain, _maxdsup, maxact);
                thrust::transform(thrust::cuda::par.on(_stream), v_dsup->begin(), v_dsup->end(), v_act->begin(), fdsupact0);
        }else{
                _F_DSUP_ACT_1 fdsupact1(_wtagain, _maxdsup);
                thrust::transform(thrust::cuda::par.on(_stream), v_dsup->begin(), v_dsup->end(), v_act->begin(), fdsupact1);
        }
}
void Hcu::update_gpu_2_3(){
        DEVICE_VECTOR(float, *v_act) = _act->mutable_gpu_vector();
        _vsum = thrust::reduce(thrust::cuda::par.on(_stream), v_act->begin(), v_act->end(), 0.0f);
}
void Hcu::update_gpu_2_4(){
        DEVICE_VECTOR(float, *v_act) = _act->mutable_gpu_vector();
        _F_DIV_VSUM fdivvsum(_vsum);
        if(_vsum>1){
                thrust::for_each(thrust::cuda::par.on(_stream), v_act->begin(), v_act->end(), fdivvsum);
        }
}
void Hcu::update_gpu_2(){

        // custom kernel
	float* ptr_act = _act->mutable_cpu_data();
	const float* ptr_dsup = _dsup->cpu_data();

	float maxdsup = ptr_dsup[0];
	for(int i=0; i<_mcu_num; i++){
		float dsup=ptr_dsup[i];
		if(dsup>maxdsup){
			maxdsup=dsup;
		}
	}
	float maxact = exp(_wtagain*maxdsup);
	float vsum=0;
	for(int j=0; j<_mcu_num; j++){
		float act = exp(_wtagain*(ptr_dsup[j]-maxdsup));
		if(maxact<1){
			act *= maxact;
		}
		vsum += act;
		*(ptr_act+j)=act;
	}
	if(vsum>1){
		for(int k=0; k<_mcu_num; k++){
			ptr_act[k] /= vsum;
		}
	}
/*
	update_halfnorm_kernel_gpu<<<1, 1, 0, _stream>>>(
		1,
		ptr_dsup,
		ptr_act,
		_mcu_num,
		_wtagain
	);
        CUDA_POST_KERNEL_CHECK;
*/
}
void Hcu::update_gpu_3(){
	/*
        CONST_DEVICE_VECTOR(float, *v_rnd) = _rnd_uniform01->gpu_vector();
        DEVICE_VECTOR(float, *v_act) = _act->mutable_gpu_vector();
        DEVICE_VECTOR(int, *v_spk) = _spike->mutable_gpu_vector();

        // generate spike
        // thrust
        _F_GEN_SPK fgenspk(_maxfqdt);
        thrust::transform(thrust::cuda::par.on(_stream), v_act->begin(), v_act->end(), v_rnd->begin()+_mcu_start, v_spk->begin()+_mcu_start, fgenspk);
*/        // custom kernel
	const float *ptr_rnd = _rnd_uniform01->gpu_data()+_mcu_start;
	const float *ptr_act = _act->gpu_data();
	int *ptr_spk = _spike->mutable_gpu_data()+_mcu_start;
        update_spike_kernel_gpu<<<GSBN_GET_BLOCKS(_mcu_num), GSBN_GET_THREADS(_mcu_num), 0, _stream>>>(
		_mcu_num,
		ptr_act,
		ptr_rnd,
		ptr_spk,
		_maxfqdt
	);
        CUDA_POST_KERNEL_CHECK; 
}

void Hcu::update_gpu(){
	LOG(INFO) << "here 0 in hcu "<< _id;
	const int *ptr_conf = static_cast<const int*>(_conf->cpu_data());
	int lginp_idx= ptr_conf[Database::IDX_CONF_STIM];
	int wmask_idx= ptr_conf[Database::IDX_CONF_GAIN_MASK];
	float wmask = (_wmask->cpu_data(wmask_idx))[_id];
	const float* ptr_epsc = _epsc->gpu_data();
	const float* ptr_bj = _bj->gpu_data();
	/*float *ptr_dsup = _dsup->mutable_gpu_data();
	const float *ptr_lginp = _lginp->gpu_data(lginp_idx)+_mcu_start;
	const float *ptr_rnd_normal = _rnd_normal->gpu_data(_mcu_start);
	
	LOG(INFO) << "here 1 in hcu "<< _id;
	CONST_DEVICE_VECTOR(float, *v_rnd) = _rnd_uniform01->gpu_vector();
	DEVICE_VECTOR(float, *v_dsup) = _dsup->mutable_gpu_vector();
	DEVICE_VECTOR(float, *v_act) = _act->mutable_gpu_vector();
	DEVICE_VECTOR(int, *v_spk) = _spike->mutable_gpu_vector();
	update_dsup_kernel_gpu<<<GSBN_GET_BLOCKS(_mcu_num), GSBN_GET_THREADS(_mcu_num), 0, _stream>>>(
		_mcu_num,
		_isp.size(),
		_mcu_num,
		ptr_epsc,
		ptr_bj,
		ptr_lginp,
		ptr_rnd_normal,
		ptr_dsup,
		_wgain,
		wmask,
		_lgbias,
		_igain,
		_taumdt
	);
	CUDA_POST_KERNEL_CHECK;	
	
	LOG(INFO) << "here 2 in hcu "<< _id;
	// halfnorm
	// thrust
	float maxdsup = thrust::reduce(thrust::cuda::par.on(_stream), v_dsup->begin(), v_dsup->end(), -1000.0f,  thrust::maximum<float>());
	float maxact = exp(_wtagain * maxdsup);
	if(maxact<1){
		_F_DSUP_ACT_0 fdsupact0(_wtagain, maxdsup, maxact);
		thrust::transform(thrust::cuda::par.on(_stream), v_dsup->begin(), v_dsup->end(), v_act->begin(), fdsupact0);
	}else{
		_F_DSUP_ACT_1 fdsupact1(_wtagain, maxdsup);
		thrust::transform(thrust::cuda::par.on(_stream), v_dsup->begin(), v_dsup->end(), v_act->begin(), fdsupact1);
	}
	
	float vsum = thrust::reduce(thrust::cuda::par.on(_stream), v_act->begin(), v_act->end(), 0.0f);
	_F_DIV_VSUM fdivvsum(vsum);
	if(vsum>1){
		thrust::for_each(thrust::cuda::par.on(_stream), v_act->begin(), v_act->end(), fdivvsum);
	}
	// custom kernel
	
	
	// generate spike
	// thrust
	_F_GEN_SPK fgenspk(_maxfqdt);
	thrust::transform(thrust::cuda::par.on(_stream), v_act->begin(), v_act->end(), v_rnd->begin()+_mcu_start, v_spk->begin()+_mcu_start, fgenspk);
	// custom kernel
	*/
}

}
}

#endif
