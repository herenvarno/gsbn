#include "gsbn/procedures/ProcUpdLazy2/Proj.hpp"


#ifndef CPU_ONLY

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

using namespace thrust;
using namespace thrust::placeholders;



namespace gsbn{
namespace proc_upd_lazy_2{

__global__ void update_full_kernel_gpu(
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
	int i=blockIdx.y*gridDim.x+blockIdx.x;
	int j=threadIdx.x;
	
	__shared__ float sh_pi;
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
		sh_pi = pi;
	}
	__syncthreads();
	
	int index = i*dim_mcu+j;
	
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
			float pi = sh_pi;
			float pj = ptr_pj[i/dim_conn*dim_mcu + j];
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
		}
	}
}

__global__ void update_j_kernel_gpu(
	int n,
	const int8_t *ptr_sj,
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
		
		pj += (ej - pj)*kp;
		ej += (zj - ej)*ke;
		zj *= (1-kzj);
		if(sj>0){
			zj += kftj;
		}
	
		if(kp){
			float bj = bgain * log(pj + eps);
			ptr_bj[idx]=bj;
		}
		ptr_pj[idx] = pj;
		ptr_ej[idx] = ej;
		ptr_zj[idx] = zj;
	}
}

__global__ void update_row_kernel_gpu(
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
	int i = blockIdx.x;
	int j = threadIdx.x;
	int row = ptr_ssi[i];
	int col = j;
	int index = row*dim_mcu+col;
	
	__shared__ float sh_pi;
	
	if(j==0){
		float pi = ptr_pi[row];
		float zi = ptr_zi[row];
		int ti = ptr_ti[row];
		int pdt = simstep - ti;
		if(pdt<=0){
			ptr_zi[row] += kfti;
			ptr_ti[row] = simstep;
		}else{
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
		sh_pi = pi;
	}
	
	__syncthreads();
	
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
		/*
		if(kp){
			float pi = sh_pi;
			float pj = ptr_pj[idx_mcu];
			wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
			ptr_wij[index] = wij;
		}else{
			wij = ptr_wij[index];
		}*/
		float pi = sh_pi;
		float pj = ptr_pj[idx_mcu];
		wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
		atomicAdd(&ptr_epsc[idx_mcu], wij);
	}
}

__global__ void update_epsc_kernel_gpu(
	int dim_conn,
	int dim_hcu,
	int dim_mcu,
	const float *ptr_siq,
	const float *ptr_wij,
	float *ptr_epsc,
	float kzi
){
	int hcu_idx = blockIdx.x;
	int mcu_idx = threadIdx.x;
	
	int offset_i = hcu_idx*dim_conn;
	int offset_j = hcu_idx*dim_mcu;
	int offset_ij = hcu_idx*dim_conn*dim_mcu;
	
	const float *ptr_siq_0 = ptr_siq+offset_i;
	const float *ptr_wij_0 = ptr_wij+offset_ij;
	float *ptr_epsc_0 = ptr_epsc+offset_j;
	
	int idx_j = hcu_idx * dim_hcu + mcu_idx;
	
	float epsc = ptr_epsc_0[idx_j] * (1-kzi);
	
	int idx_ij = mcu_idx;
	for(int i=0; i<dim_conn; i++){
		if(ptr_siq_0[i]>0){
			epsc += ptr_wij_0[idx_ij];
		}
		idx_ij += dim_mcu;
	}
	
	ptr_epsc_0[idx_j] = epsc;
	
}

__global__ void update_col_kernel_gpu(
	int dim_conn,
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

	int i = blockIdx.x;
	int j = threadIdx.x;

	int row = ptr_ssj[j]/dim_mcu*dim_conn+i;
	if(ptr_ii[row]<0){
		return;
	}
	int col = ptr_ssj[j]%dim_mcu;
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


void Proj::update_full_gpu(float prn, int simstep){
	float *ptr_pi = _pi->mutable_gpu_data();
	float *ptr_ei = _ei->mutable_gpu_data();
	float *ptr_zi = _zi->mutable_gpu_data();
	int *ptr_ti = _ti->mutable_gpu_data();
	const float *ptr_pj = _pj->gpu_data();
	float *ptr_pij = _pij->mutable_gpu_data();
	float *ptr_eij = _eij->mutable_gpu_data();
	float *ptr_zi2 = _zi2->mutable_gpu_data();
	float *ptr_zj2 = _zj2->mutable_gpu_data();
	int *ptr_tij = _tij->mutable_gpu_data();
	float *ptr_wij = _wij->mutable_gpu_data();
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
		_taupdt*prn,
		_tauedt,
		_tauzidt,
		_tauzjdt,
		_wgain,
		_eps,
		_eps2
	);
	CUDA_POST_KERNEL_CHECK;
}

void Proj::update_j_gpu(){
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float prn = ptr_conf[Database::IDX_CONF_PRN];
	float *ptr_pj = _pj->mutable_gpu_data();
	float *ptr_ej = _ej->mutable_gpu_data();
	float *ptr_zj = _zj->mutable_gpu_data();
	float *ptr_bj = _bj->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_epsc = _epsc->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
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
		_eps
	);
	CUDA_POST_KERNEL_CHECK;
}

__global__ void update_siq_kernel_gpu(
	int cycle,
	int n,
	int dim_hcu,
	int dim_mcu,
	const int *ptr_ii,
	const int *ptr_di,
	const int8_t *ptr_si,
	int *ptr_qi,
	int8_t *ptr_siq
){
	const int8_t *ptr_si_0 = ptr_si;
	int8_t *ptr_siq_0 = ptr_siq;
	int ld = dim_hcu * dim_mcu;
	CUDA_KERNEL_LOOP(i, n){
		int ii=ptr_ii[i];
		int di=ptr_di[i];
		int32_t qi = ptr_qi[i];
		for(int c=0; c<cycle; c++){
			if(ii>=0){
				qi >>= 1;
				ptr_siq_0[i] = float(qi & 0x01);
				
				int8_t spk = ptr_si_0[ii];
				if(spk>0){
					qi |= (0x01 << di);
				}
			}
			ptr_si_0 += ld;
			ptr_siq_0 += ld;
		}
		ptr_qi[i]=qi;
	}
}

void Proj::update_ss_gpu(int cycle){
	const int *ptr_ii = _ii->gpu_data();
	const int *ptr_di = _di->gpu_data();
	const int8_t *ptr_si = _si->gpu_data();
	int *ptr_qi = _qi->mutable_gpu_data();
	int8_t *ptr_siq = _siq->mutable_gpu_data();
	
	update_siq_kernel_gpu<<<GSBN_GET_BLOCKS(_dim_hcu* _dim_conn), GSBN_GET_THREADS(_dim_hcu* _dim_conn), 0, _stream>>>(
		cycle,
		_dim_hcu * _dim_conn,
		_dim_hcu,
		_dim_mcu,
		ptr_ii,
		ptr_di,
		ptr_si,
		ptr_qi,
		ptr_siq
	);
	CUDA_POST_KERNEL_CHECK;
	
	/*
	DEVICE_VECTOR_ITERATOR(int, it);
	
	// get active in spike
	CONST_DEVICE_VECTOR(float, *v_siq) = _siq->gpu_vector();
	DEVICE_VECTOR(int, *v_ssi) = _ssi->mutable_gpu_vector();
	v_ssi->resize(v_siq->size());
	it = copy_if(
		thrust::cuda::par.on(_stream),
		make_counting_iterator<int>(0),
		make_counting_iterator<int>(v_siq->size()),
		v_siq->begin(),
		v_ssi->begin(),
		_1>0);
	v_ssi->resize(thrust::distance(v_ssi->begin(), it));
	
	// get active out spike
	CONST_DEVICE_VECTOR(int8_t, *v_sj) = _sj->gpu_vector();
	DEVICE_VECTOR(int, *v_ssj) = _ssj->mutable_gpu_vector();
	v_ssj->resize(v_sj->size());
	it = copy_if(
		thrust::cuda::par.on(_stream),
		make_counting_iterator<int>(0),
		make_counting_iterator<int>(v_sj->size()),
		v_sj->begin(),
		v_ssj->begin(),
		_1>0);
	v_ssj->resize(thrust::distance(v_ssj->begin(), it));
	*/
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
	
	float *ptr_pi = _pi->mutable_gpu_data();
	float *ptr_ei = _ei->mutable_gpu_data();
	float *ptr_zi = _zi->mutable_gpu_data();
	int *ptr_ti = _ti->mutable_gpu_data();
	const float *ptr_pj = _pj->gpu_data();
	float *ptr_pij = _pij->mutable_gpu_data();
	float *ptr_eij = _eij->mutable_gpu_data();
	float *ptr_zi2 = _zi2->mutable_gpu_data();
	float *ptr_zj2 = _zj2->mutable_gpu_data();
	int *ptr_tij = _tij->mutable_gpu_data();
	float *ptr_wij = _wij->mutable_gpu_data();
	float *ptr_epsc = _epsc->mutable_gpu_data()+ _proj_in_pop * _dim_hcu * _dim_mcu;
	const int8_t *ptr_siq = _siq->gpu_data();
	
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
		_eps2
	);
	CUDA_POST_KERNEL_CHECK;
	/*
	update_epsc_kernel_gpu<<<_dim_hcu, _dim_mcu, 0, _stream>>>(
		_dim_conn,
		_dim_hcu,
		_dim_mcu,
		ptr_siq,
		ptr_wij,
		ptr_epsc,
		_tauzidt
	);
	CUDA_POST_KERNEL_CHECK;
	*/
	/*
	float alpha = 0;
	float beta = 1-_tauzidt;
	cublasSetStream(CUBLAS_HANDLE, _stream);
	for(int i=0; i<_dim_hcu; i++){
		CUBLAS_CHECK(
			cublasSgemv(
				CUBLAS_HANDLE,
				CUBLAS_OP_N,
				_dim_mcu,
				_dim_conn,
				&alpha,
				ptr_wij+(_dim_conn*_dim_mcu)*i,
				_dim_mcu,
				ptr_siq+(_dim_conn)*i,
				1,
				&beta,
				ptr_epsc+(_dim_mcu)*i,
				1
			)
		);
	}*/
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
	
	float *ptr_pij = _pij->mutable_gpu_data();
	float *ptr_eij = _eij->mutable_gpu_data();
	float *ptr_zi2 = _zi2->mutable_gpu_data();
	float *ptr_zj2 = _zj2->mutable_gpu_data();
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
		_kftj
	);
	CUDA_POST_KERNEL_CHECK;
}

__global__ void update_zep_kernel_gpu(
	int cycle,
	int dim_conn,
	int dim_hcu,
	int dim_mcu,
	const int8_t *ptr_siq,
	const int8_t *ptr_sj,
	float *ptr_pi,
	float *ptr_ei,
	float *ptr_zi,
	int *ptr_ti,
	float *ptr_pj,
	float *ptr_ej,
	float *ptr_zj,
	float *ptr_epsc,
	float *ptr_bj,
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
	float kfti,
	float kftj,
	float wgain,
	float bgain,
	float eps,
	float eps2
){
	
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
	float *ptr_zi2_0 = ptr_zi2+offset_ij;
	float *ptr_zj2_0 = ptr_zj2+offset_ij;
	float *ptr_bj_0 = ptr_bj+offset_j;
	float *ptr_epsc_0 = ptr_epsc+offset_j;
	float *ptr_wij_0 = ptr_wij+offset_ij;
	int *ptr_ti_0 = ptr_ti+offset_i;
	int *ptr_tij_0 = ptr_tij+offset_ij;
	const int8_t *ptr_siq_0 = ptr_siq+offset_i;
	const int8_t *ptr_sj_0 = ptr_sj+offset_j;
	
	
	float pj = ptr_pj_0[mcu_idx];
	float ej = ptr_ej_0[mcu_idx];
	float zj = ptr_zj_0[mcu_idx];
	float epsc = ptr_epsc_0[mcu_idx] * (1-kzi);
	
	// update j
	const int8_t *ptr_sj_0_0 = ptr_sj_0;
	for(int c=0; c<cycle; c++){
		int8_t sj = ptr_sj_0_0[mcu_idx];
		pj += (ej - pj)*kp;
		ej += (zj - ej)*ke;
		zj *= (1-kzj);
		if(sj>0){
			zj += kftj;
		}
		ptr_sj_0_0 += (dim_hcu * dim_mcu);
	}
	if(kp){
		float bj = bgain * log(pj + eps);
		ptr_bj_0[mcu_idx] = bj;
	}
	ptr_pj_0[mcu_idx] = pj;
	ptr_ej_0[mcu_idx] = ej;
	ptr_zj_0[mcu_idx] = zj;
	ptr_epsc_0[mcu_idx] = epsc;
	
	// update i
	
	int r=0;
	while(r<dim_conn){
		int index_i = r+mcu_idx;
		if(index_i>=dim_conn){
			break;
		}
		float pi;
		float ei;
		float zi;
		int ti;
		bool changed = false;
		const int8_t *ptr_siq_0_0 = ptr_siq_0;
		for(int c=0; c<cycle; c++){
			int8_t si = ptr_siq_0_0[index_i];
			if(si>0){
				if(!changed){
					pi = ptr_pi_0[index_i];
					ei = ptr_ei_0[index_i];
					zi = ptr_zi_0[index_i];
					ti = ptr_ti_0[index_i];
				}
				changed = true;
				int pdt = ti - simstep;
				if(pdt <= 0){
					zi += kfti;
					ti = simstep;
				}else{
					pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
						(ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
						((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
						(ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
					ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
						(ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
					zi = zi*exp(-kzi*pdt)+kfti;
					ti = simstep;
				}
				
			}
			ptr_siq_0_0 += (dim_hcu *dim_conn);
		}
		if(changed){
			ptr_pi_0[index_i] = pi;
			ptr_ei_0[index_i] = ei;
			ptr_zi_0[index_i] = zi;
			ptr_ti_0[index_i] = ti;
		}
		
		r += dim_mcu;
	}
	
	r=0;
	while(r<dim_conn){
		int index_i = r+mcu_idx;
		if(index_i>=dim_conn){
			break;
		}
		int index_ij = r*dim_mcu+mcu_idx;
		bool changed = false;
		const int8_t *ptr_siq_0_0 = ptr_siq_0;
		const int8_t *ptr_sj_0_0 = ptr_sj_0;
		
		float pij;
		float eij;
		float zi2;
		float zj2;
		int tij;
		for(int c=0; c<cycle; c++){
			int8_t si = ptr_siq_0_0[index_i];
			int8_t sj = ptr_sj_0_0[mcu_idx];
			if(si <= 0 && sj <= 0){
				continue;
			}
			
			if(!changed){
				pij = ptr_pij_0[index_ij];
				eij = ptr_eij_0[index_ij];
				zi2 = ptr_zi2_0[index_ij];
				zj2 = ptr_zj2_0[index_ij];
				tij = ptr_tij_0[index_ij];
			}
			int pdt = tij - simstep;
			if(pdt<0){
				if(si>0){
					zi2 += kfti;
				}
				if(sj > 0){
					zj2 += kftj;
				}
				tij = simstep;
			}else{
				if(si <=0){
					zi2 = ptr_zi2_0[index_ij];
				}
				if(sj <=0){
					zj2 = ptr_zj2_0[index_ij];
				}
				
				pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2)/(ke - kp) -
					(ke*kp*zi2*zj2)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
					((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi2*zj2))/(ke - kp) -
					(ke*kp*zi2*zj2*exp(kp*pdt - kzi*pdt - kzj*pdt))/
					(kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
				eij = (eij + (ke*zi2*zj2)/(kzi - ke + kzj))/exp(ke*pdt) -
					(ke*zi2*zj2)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
				zi2 = zi2*exp(-kzi*pdt);
				if(si>0){
					zi2 += kfti;
				}
				zj2 = zj2*exp(-kzj*pdt);
				if(sj>0){
					zj += kftj;
				}
				tij = simstep;
			}
			changed = true;
			ptr_siq_0_0 += dim_hcu * dim_conn;
			ptr_sj_0_0 += dim_hcu * dim_mcu;
			
			if(c==cycle-1){
				float wij;
				if(kp){
					float pi = ptr_pi[index_i];
					wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
					ptr_wij_0[index_ij] = wij;
				}else{
					wij = ptr_wij_0[index_ij];
				}
				epsc += wij;
			}
		}
		if(changed){
			ptr_pij_0[index_ij] = pij;
			ptr_eij_0[index_ij] = eij;
			ptr_zi2_0[index_ij] = zi2;
			ptr_zj2_0[index_ij] = zj2;
			ptr_tij_0[index_ij] = tij;
		}
		
		r+=1;
	}
}
/*
void Proj::update_siq_gpu(){
	const int *ptr_ii = _ii->gpu_data();
	const int *ptr_di = _di->gpu_data();
	const int8_t *ptr_si = _si->gpu_data();
	int *ptr_qi = _qi->mutable_gpu_data();
	int8_t *ptr_siq = _siq->mutable_gpu_data();
	
	update_siq_kernel_gpu<<<GSBN_GET_BLOCKS(_dim_hcu* _dim_conn), GSBN_GET_THREADS(_dim_hcu* _dim_conn), 0, _stream>>>(
		_dim_hcu * _dim_conn,
		ptr_ii,
		ptr_di,
		ptr_si,
		ptr_qi,
		ptr_siq
	);
	CUDA_POST_KERNEL_CHECK;
}
*/
void Proj::update_zep_gpu(int cycle, float prn, int simstep){
	float *ptr_pi = _pi->mutable_gpu_data();
	float *ptr_ei = _ei->mutable_gpu_data();
	float *ptr_zi = _zi->mutable_gpu_data();
	int *ptr_ti = _ti->mutable_gpu_data();
	float *ptr_pj = _pj->mutable_gpu_data();
	float *ptr_ej = _ej->mutable_gpu_data();
	float *ptr_zj = _zj->mutable_gpu_data();
	float *ptr_bj = _bj->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_epsc = _epsc->mutable_gpu_data()+_proj_in_pop*_dim_hcu*_dim_mcu;
	float *ptr_pij = _pij->mutable_gpu_data();
	float *ptr_eij = _eij->mutable_gpu_data();
	float *ptr_zi2 = _zi2->mutable_gpu_data();
	float *ptr_zj2 = _zj2->mutable_gpu_data();
	int *ptr_tij = _tij->mutable_gpu_data();
	float *ptr_wij = _wij->mutable_gpu_data();
	const int8_t *ptr_siq = _siq->gpu_data();
	const int8_t *ptr_sj = _sj->gpu_data();
	
	CUDA_CHECK(cudaFuncSetCacheConfig(update_zep_kernel_gpu, cudaFuncCachePreferShared));
	update_zep_kernel_gpu<<<_dim_hcu, _dim_mcu, 3*_dim_mcu*sizeof(float), _stream>>>(
		cycle,
		_dim_conn,
		_dim_hcu,
		_dim_mcu,
		ptr_siq,
		ptr_sj,
		ptr_pi,
		ptr_ei,
		ptr_zi,
		ptr_ti,
		ptr_pj,
		ptr_ej,
		ptr_zj,
		ptr_epsc,
		ptr_bj,
		ptr_pij,
		ptr_eij,
		ptr_zi2,
		ptr_zj2,
		ptr_tij,
		ptr_wij,
		simstep,
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
