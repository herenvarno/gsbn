#include "gsbn/Net.hpp"

namespace gsbn{

void Net::update_phase_0_gpu(){
	int size = _rnd_uniform01->height();
	_rnd.gen_uniform01_gpu(_rnd_uniform01->mutable_gpu_data(), _rnd_uniform01->height());
        
	int h_hcu = _hcu->height();
	int idx=0;
	for(int i=0; i<h_hcu; i++){
		int mcu_num = static_cast<const int *>(_hcu->cpu_data(i))[Database::IDX_HCU_MCU_NUM];
		float snoise = static_cast<const float *>(_hcu->cpu_data(i))[Database::IDX_HCU_SNOISE];
		_rnd.gen_uniform01_cpu(_rnd_normal->mutable_gpu_data(idx), mcu_num, 0, snoise);
		idx += mcu_num;
	}
}

/*
 * Phase 1: update Pj, Ej, Zj, EPSC, DSUP
 * loop for #mcu times. inside each loop, it calculates #hcu_isp elements which
 * belongs to the same MCU.
 */
__global__ void update_kernel_phase_1_gpu(
	int n,
	const void *ptr_mcu, int w_mcu,
	const void *ptr_addr, int w_addr,
	const void *ptr_hcu, int w_hcu,
	const void *ptr_hcu_isp, int w_hcu_isp,
	const void *ptr_proj, int w_proj,
	const void *ptr_stim, int w_stim,
	const void *ptr_rnd_normal, int w_rnd_normal,
	void *ptr_j_array, int w_j_array,
	void *ptr_epsc, int w_epsc,
	void *ptr_sup, int w_sup,
	float pdt, float prn){
	
	int idx;	
	CUDA_KERNEL_LOOP(idx, n) {
		int hcu_idx = static_cast<const int*>(ptr_addr+idx*w_addr)[Database::IDX_ADDR_HCU];
		const int *ptr_hcu_data = static_cast<const int*>(ptr_hcu+hcu_idx*w_hcu);
		int hcu_isp_idx = ptr_hcu_data[Database::IDX_HCU_ISP_INDEX];
		int hcu_isp_num = ptr_hcu_data[Database::IDX_HCU_ISP_NUM];
		const float *ptr_hcu_data0 = static_cast<const float*>(ptr_hcu+hcu_idx*w_hcu);
		float taumdt = ptr_hcu_data0[Database::IDX_HCU_TAUMDT];
		float snoise = ptr_hcu_data0[Database::IDX_HCU_SNOISE];
		float igain = ptr_hcu_data0[Database::IDX_HCU_IGAIN];
		float lgbias = ptr_hcu_data0[Database::IDX_HCU_LGBIAS];
		float wgain = ptr_hcu_data0[Database::IDX_HCU_WGAIN];
		float wsup=0;
		int j_array_idx=static_cast<const int*>(ptr_mcu+idx*w_mcu)[Database::IDX_MCU_J_ARRAY_INDEX];
		for(int i=0; i<hcu_isp_num; i++){
			int proj_idx = static_cast<const int*>(ptr_hcu_isp+(hcu_isp_idx+i)*w_hcu_isp)[Database::IDX_HCU_ISP_VALUE];
			const float *ptr_proj_data = static_cast<const float*>(ptr_proj+proj_idx*w_proj);
			float kzi = ptr_proj_data[Database::IDX_PROJ_TAUZIDT];
			float kzj = ptr_proj_data[Database::IDX_PROJ_TAUZJDT];
			float ke = ptr_proj_data[Database::IDX_PROJ_TAUEDT];
			float kp = ptr_proj_data[Database::IDX_PROJ_TAUPDT]*prn;
			float eps = ptr_proj_data[Database::IDX_PROJ_EPS];
			float bgain = ptr_proj_data[Database::IDX_PROJ_BGAIN];
			float prntaupdt = kp;
		
			float *ptr_j_array_data = static_cast<float *>(ptr_j_array+(j_array_idx+i)*w_j_array);
			float pj = ptr_j_array_data[Database::IDX_J_ARRAY_PJ];
			float ej = ptr_j_array_data[Database::IDX_J_ARRAY_EJ];
			float zj = ptr_j_array_data[Database::IDX_J_ARRAY_ZJ];
			float *ptr_epsc_data = static_cast<float *>(ptr_epsc+(j_array_idx+i)*w_epsc);
			float epsc = ptr_epsc_data[Database::IDX_EPSC_VALUE];
			float bj;
		
			pj = (pj - ((ej*kp*kzj - ej*ke*kp + ke*kp*zj)/(ke - kp) +
				                (ke*kp*zj)/(kp - kzj))/(ke - kzj))/exp(kp*pdt) +
				    ((exp(kp*pdt - ke*pdt)*(ej*kp*kzj - ej*ke*kp + ke*kp*zj))/(ke - kp) +
				     (ke*kp*zj*exp(kp*pdt - kzj*pdt))/(kp - kzj))/(exp(kp*pdt)*(ke - kzj));
			ej = (ej - (ke*zj)/(ke - kzj))/exp(ke*pdt) +
				    (ke*zj*exp(ke*pdt - kzj*pdt))/(exp(ke*pdt)*(ke - kzj));
			zj = zj*exp(-kzj*pdt);
			epsc *= (1-kzi);
		
			if(prntaupdt==0){
				bj = ptr_j_array_data[Database::IDX_J_ARRAY_BJ];
			}else{
				bj = bgain * log(pj + eps);
				ptr_j_array_data[Database::IDX_J_ARRAY_BJ]=bj;
			}
			wsup += bj + epsc;
		
			ptr_j_array_data[Database::IDX_J_ARRAY_PJ] = pj;
			ptr_j_array_data[Database::IDX_J_ARRAY_EJ] = ej;
			ptr_j_array_data[Database::IDX_J_ARRAY_ZJ] = zj;
			ptr_epsc_data[Database::IDX_EPSC_VALUE] = epsc;
		}
	
		const float *ptr_stim_data=static_cast<const float *>(ptr_stim+idx*w_stim); //FIXME : CHECK INDEX OF STIM.
		const float *ptr_rnd_normal_data=static_cast<const float *>(ptr_rnd_normal+idx*w_rnd_normal);
		//float sup = lgbias + igain * ptr_stim_data[Database::IDX_STIM_VALUE] + Random::gen_normal(0, snoise);
		float sup = lgbias + igain * ptr_stim_data[Database::IDX_STIM_VALUE] + ptr_rnd_normal_data[Database::IDX_RND_NORMAL_VALUE];
		sup += wgain * wsup;
	
		float *ptr_sup_data = static_cast<float *>(ptr_sup+idx*w_sup);
		float dsup=ptr_sup_data[Database::IDX_SUP_DSUP];
		ptr_sup_data[Database::IDX_SUP_DSUP] += (sup - dsup) * taumdt;
	
	}
}
void Net::update_phase_1_gpu(){
	int h_mcu = _mcu->height();
	int w_mcu = _mcu->width();
	const void *ptr_mcu = _mcu->gpu_data();
	int w_addr = _addr->width();
	const void *ptr_addr = _addr->gpu_data();
	int w_hcu = _hcu->width();
	const void *ptr_hcu = _hcu->gpu_data();
	int w_hcu_isp = _hcu_isp->width();
	const void *ptr_hcu_isp = _hcu_isp->gpu_data();
	int w_proj = _proj->width();
	const void *ptr_proj = _proj->gpu_data();
	int w_stim = _stim->width();
	const void *ptr_stim = _stim->gpu_data();
	int w_rnd_normal = _rnd_normal->width();
	const void *ptr_rnd_normal = _rnd_normal->gpu_data();
	int w_j_array = _j_array->width();
	void *ptr_j_array = _j_array->mutable_gpu_data();
	int w_epsc = _epsc->width();
	void *ptr_epsc = _epsc->mutable_gpu_data();
	int w_sup = _sup->width();
	void *ptr_sup = _sup->mutable_gpu_data();
	
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float dt=ptr_conf[Database::IDX_CONF_DT];
	float prn=ptr_conf[Database::IDX_CONF_PRN];
	
	update_kernel_phase_1_gpu<<<GSBN_GET_BLOCKS(h_mcu), GSBN_CUDA_NUM_THREADS>>>(
		h_mcu,
		ptr_mcu, w_mcu,
		ptr_addr, w_addr,
		ptr_hcu, w_hcu,
		ptr_hcu_isp, w_hcu_isp,
		ptr_proj, w_proj,
		ptr_stim, w_stim,
		ptr_rnd, w_rnd,
		ptr_j_array, w_j_array,
		ptr_epsc, w_epsc,
		ptr_sup, w_sup,
		dt, prn);
  CUDA_POST_KERNEL_CHECK;
}

/*
 * Phase 2: update dsup
 * No need for timestamp
 */
/*
void update_kernel_phase_2_gpu(
	int idx,
	const void *ptr_mcu, int w_mcu,
	const void *ptr_epsc, int w_epsc,
	const void *ptr_j_array, int w_j_array,
	const void *ptr_stim, int w_stim,
	void *ptr_sup, int w_sup,
	float lgbias, float igain, float snoise, float taumdt,
	float wgain, float bgain,
	float eps){

	const int *ptr_mcu_data = static_cast<const int *>(ptr_mcu+idx*w_mcu);
	int j_array_idx = ptr_mcu_data[Database::IDX_MCU_J_ARRAY_INDEX];
	int j_array_num = ptr_mcu_data[Database::IDX_MCU_J_ARRAY_NUM];
	
	float *ptr_sup_data = static_cast<float *>(ptr_sup+idx*w_sup);
	float dsup = ptr_sup_data[Database::IDX_SUP_DSUP];
	float wsup=0;
	for(int i=0; i<j_array_num; i++){
		const float *ptr_j_array_data=static_cast<const float *>(ptr_j_array+(i+j_array_idx)*w_j_array);
		float pj=ptr_j_array_data[Database::IDX_J_ARRAY_PJ];
		const float *ptr_epsc_data=static_cast<const float *>(ptr_epsc+(i+j_array_idx)*w_epsc);
		float epsc=ptr_epsc_data[Database::IDX_EPSC_VALUE];
		float bj = bgain * log(pj + eps);
		wsup += bj + epsc;
	}
	const float *ptr_stim_data=static_cast<const float *>(ptr_stim+idx*w_stim); //FIXME : CHECK INDEX OF STIM.
	float sup = lgbias + igain * ptr_stim_data[0] + Random::gen_normal(0, snoise);
	sup += wgain * wsup;
	dsup += (sup - dsup) * taumdt;
	ptr_sup_data[Database::IDX_SUP_DSUP]=dsup;
}
void Net::update_phase_2_gpu(int stim_offset){
	int h_sup=_sup->height();
	int w_sup=_sup->width();
	void *ptr_sup=_sup->mutable_cpu_data();
	int w_mcu = _mcu->width();
	const void *ptr_mcu = _mcu->cpu_data();
	int w_epsc = _epsc->width();
	const void *ptr_epsc = _epsc->cpu_data();
	int w_j_array = _j_array->width();
	const void *ptr_j_array = _j_array->cpu_data();
	int w_stim = _stim->width();
	const void *ptr_stim = _stim->cpu_data(stim_offset);

	for(int i=0; i<h_sup; i++){
		update_kernel_phase_2_gpu(
			i,
			ptr_mcu, w_mcu,
			ptr_epsc, w_epsc,
			ptr_j_array, w_j_array,
			ptr_stim, w_stim,
			ptr_sup, w_sup,
			_lgbias, _igain, _snoise, _taumdt,
			_wgain, _bgain,
			_eps);
	}
}
*/

/*
 * Phase 3: halfnormlize
 * No need for timestamp
 */
__global__ void update_kernel_phase_3_gpu(
	int n, 
	const void *ptr_hcu, int w_hcu,
	void *ptr_sup, int w_sup){

	int idx;
	CUDA_KERNEL_LOOP(idx, n) {
		const int *ptr_hcu_data = static_cast<const int *>(ptr_hcu+idx*w_hcu);
		int mcu_idx = ptr_hcu_data[Database::IDX_HCU_MCU_INDEX];
		int mcu_num = ptr_hcu_data[Database::IDX_HCU_MCU_NUM];
		const float *ptr_hcu_data0 = static_cast<const float *>(ptr_hcu+idx*w_hcu);
		float wtagain = ptr_hcu_data0[Database::IDX_HCU_WTAGAIN];
	
		float maxdsup = static_cast<float *>(ptr_sup+mcu_idx*w_sup)[Database::IDX_SUP_DSUP];
		for(int i=0; i<mcu_num; i++){
			float dsup=static_cast<float *>(ptr_sup+(mcu_idx+i)*w_sup)[Database::IDX_SUP_DSUP];
			if(dsup>maxdsup){
				maxdsup=dsup;
			}
		}
		float maxact = exp(wtagain*maxdsup);
		float vsum=0;
		for(int i=0; i<mcu_num; i++){
			float *ptr_sup_data = static_cast<float *>(ptr_sup+(mcu_idx+i)*w_sup);
			float dsup = ptr_sup_data[Database::IDX_SUP_DSUP];
			float act = exp(wtagain*(dsup-maxdsup));
			act = maxact<1?act*maxact:act;
			vsum += act;
			ptr_sup_data[Database::IDX_SUP_ACT]=act;
		}
	
		if(vsum>1){
			for(int i=0; i<mcu_num; i++){
				static_cast<float *>(ptr_sup+(mcu_idx+i)*w_sup)[Database::IDX_SUP_ACT]/=vsum;
			}
		}
	}
}
void Net::update_phase_3_gpu(){
	int h_hcu=_hcu->height();
	int w_hcu=_hcu->width();
	const void *ptr_hcu=_hcu->gpu_data();
	int w_sup=_sup->width();
	void *ptr_sup=_sup->mutable_gpu_data();

	update_kernel_phase_3_gpu<<<GSBN_GET_BLOCKS(h_hcu), GSBN_CUDA_NUM_THREADS>>>(
		h_hcu,
		ptr_hcu, w_hcu,
		ptr_sup, w_sup);
  CUDA_POST_KERNEL_CHECK;
}

/*
 * Phase 4: generate spike
 * No need for timestamp
 */
__global__ void update_kernel_phase_4_gpu(
	int n,
	const void *ptr_addr, int w_addr,
	const void *ptr_hcu, int w_hcu,
	const void *ptr_sup, int w_sup,
	const void *ptr_rnd_uniform01, int w_rnd_uniform01,
	void *ptr_spk, int w_spk){
	
	int idx;
	CUDA_KERNEL_LOOP(idx, n) {
		int hcu_idx = static_cast<const int*>(ptr_addr+idx*w_addr)[Database::IDX_ADDR_HCU];
		float maxfqdt = static_cast<const float*>(ptr_hcu+hcu_idx*w_hcu)[Database::IDX_HCU_MAXFQDT];
		unsigned char *ptr_spk_data = static_cast<unsigned char *>(ptr_spk+idx*w_spk);
		float act = static_cast<const float *>(ptr_sup+idx*w_sup)[Database::IDX_SUP_ACT];
		float randnum = static_cast<const float *>(ptr_rnd_uniform01+idx*w_rnd)[Database::IDX_RND_UNIFORM01_VALUE];
		ptr_spk_data[Database::IDX_SPK_VALUE]= randnum < act*maxfqdt;
	}
}
void Net::update_phase_4_gpu(){
	int h_spk=_spk->height();
	int w_spk=_spk->width();
	void *ptr_spk=_spk->mutable_gpu_data();
	int w_addr = _addr->width();
	const void *ptr_addr = _addr->gpu_data();
	int w_hcu = _hcu->width();
	const void *ptr_hcu = _hcu->gpu_data();
	int w_sup=_sup->width();
	const void *ptr_sup=_sup->gpu_data();
	int w_rnd_uniform01=_rnd_uniform01->width();
	const void *ptr_rnd_uniform01=_rnd_uniform01->gpu_data();
	
	update_kernel_phase_4_gpu<<<GSBN_GET_BLOCKS(h_spk), GSBN_CUDA_NUM_THREADS>>>(
		h_spk,
		ptr_addr, w_addr,
		ptr_hcu, w_hcu,
		ptr_sup, w_sup,
		ptr_rnd_uniform01, w_rnd_uniform01,
		ptr_spk, w_spk);
	CUDA_POST_KERNEL_CHECK;
}

/*
 * Phase 5: generate short spike list, tmp table
 * No need for timestamp
 */
__global__ update_kernel_phase_5_gpu(
	int n,
	const void *ptr_spk, int w_spk,
	void *ptr_tmp1, int w_tmp1,
	int *tmp1_size){

	CUDA_KERNEL_LOOP(idx, n) {
		__shared__ int size;
		
		__syncthreads();
		
		if(idx==0){
			size = 0;
		}
		
		__syncthreads();
		
		const unsigned char spike = static_cast<const unsigned char *>(ptr_spk+idx*w_spk)[Database::IDX_SPK_VALUE];
		if(spike!=0){
			int location = atomaticInc(size);
			static_cast<int *>(ptr_tmp1+location*w_tmp1)[Database::IDX_TMP1_MCU_IDX]=idx;
		}
		
		__syncthreads();
		
		if(idx==0){
			*tmp1_size = size;
		}
		
}

void Net::update_phase_5_gpu(){	
	int h_spk=_spk->height();
	int w_spk=_spk->width();
	void *ptr_spk=_spk->gpu_data();
	int w_tmp1=_tmp1->width();
	void *ptr_tmp1=_tmp1->mutable_gpu_data();
	
	__device__ int list_size_gpu;
	_tmp1->reset();
	update_kernel_phase_5_gpu<<<1, GSBN_CUDA_NUM_THREADS>>>(
		h_spk,
		ptr_spk, w_spk,
		ptr_tmp1, w_tmp1,
		&list_size_gpu);
	CUDA_POST_KERNEL_CHECK;
	int list_size_cpu;
	MemBlock::memcpy_gpu_to_cpu(&list_size_cpu, &list_size_gpu, sizeof(int));
	_tmp1->set_rows(list_size_cpu);
}

/*
 * Phase 6: increase Zj
 * No need for timestamp
 */
__global__ void update_kernel_phase_6_gpu(
	int n, 
	const void *ptr_tmp1, int w_tmp1,
	const void *ptr_mcu, int w_mcu,
	const void *ptr_addr, int w_addr,
	const void *ptr_hcu, int w_hcu,
	const void *ptr_hcu_isp, int w_hcu_isp,
	const void *ptr_proj, int w_proj,
	void *ptr_j_array, int w_j_array){
	
	int idx_tmp1;
	CUDA_KERNEL_LOOP(idx_tmp1, n) {
		int mcu_idx = static_cast<const int *>(ptr_tmp1+idx_tmp1*w_tmp1)[Database::IDX_TMP1_MCU_IDX];
		int hcu_idx = static_cast<const int *>(ptr_addr+mcu_idx*w_addr)[Database::IDX_ADDR_HCU];
		int hcu_isp_idx = static_cast<const int *>(ptr_hcu+hcu_idx*w_hcu)[Database::IDX_HCU_ISP_INDEX];
		const int *ptr_mcu_data = static_cast<const int *>(ptr_mcu+mcu_idx*w_mcu);
		int idx_j_array = ptr_mcu_data[Database::IDX_MCU_J_ARRAY_INDEX];
		int num_j_array = ptr_mcu_data[Database::IDX_MCU_J_ARRAY_NUM];
		for(int i=0; i<num_j_array; i++){
			int proj_idx = static_cast<const int *>(ptr_hcu_isp+(hcu_isp_idx+i)*w_hcu_isp)[Database::IDX_HCU_ISP_VALUE];
			float kftj = static_cast<const float *>(ptr_proj+proj_idx*w_proj)[Database::IDX_PROJ_KFTJ];
			float *ptr_j_array_data = static_cast<float *>(ptr_j_array+(idx_j_array+i)*w_j_array);
			ptr_j_array_data[Database::IDX_J_ARRAY_ZJ]+=kftj;
		}
	}
}
void Net::update_phase_6_gpu(){
	int h_tmp1=_tmp1->height();
	int w_tmp1=_tmp1->width();
	if(h_tmp1<=0)
		return;
	const void *ptr_tmp1=_tmp1->gpu_data();
	int w_mcu=_mcu->width();
	const void *ptr_mcu=_mcu->gpu_data();
	int w_addr = _addr->width();
	const void *ptr_addr = _addr->gpu_data();
	int w_hcu = _hcu->width();
	const void *ptr_hcu = _hcu->gpu_data();
	int w_hcu_isp = _hcu_isp->width();
	const void *ptr_hcu_isp = _hcu_isp->gpu_data();
	int w_proj = _proj->width();
	const void *ptr_proj = _proj->gpu_data();
	int w_j_array = _j_array->width();
	void * ptr_j_array = _j_array->mutable_gpu_data();
	
	update_kernel_phase_6_gpu<<<GSBN_GET_BLOCKS(h_tmp1), GSBN_CUDA_NUM_THREADS>>>(
		h_tmp1,
		ptr_tmp1, w_tmp1,
		ptr_mcu, w_mcu,
		ptr_addr, w_addr,
		ptr_hcu, w_hcu,
		ptr_hcu_isp, w_hcu_isp,
		ptr_proj, w_proj,
		ptr_j_array, w_j_array);
	CUDA_POST_KERNEL_CHECK;
}










/*
 * Phase 7: Scan and generate short coming spike list.
 */
void Net::update_phase_7_gpu(){
	int h_conn=_conn->height();
	_tmp2->reset();
	for(int i=0; i<h_conn; i++){
		int *ptr_conn = static_cast<int *>(_conn->mutable_cpu_data(i, 0));
		int queue=ptr_conn[Database::IDX_CONN_QUEUE];
		if(queue & 0x01){
			int *ptr_tmp2 = static_cast<int *>(_tmp2->expand(1));
			ptr_tmp2[Database::IDX_TMP2_CONN]=i;
			ptr_tmp2[Database::IDX_TMP2_DEST_HCU]=ptr_conn[Database::IDX_CONN_DEST_HCU];
			ptr_tmp2[Database::IDX_TMP2_SUBPROJ]=ptr_conn[Database::IDX_CONN_SUBPROJ];
			ptr_tmp2[Database::IDX_TMP2_PROJ]=ptr_conn[Database::IDX_CONN_PROJ];
			ptr_tmp2[Database::IDX_TMP2_IJ_MAT_INDEX]=ptr_conn[Database::IDX_CONN_IJ_MAT_INDEX];
		}
		ptr_conn[Database::IDX_CONN_QUEUE] = queue >> 1;
	}
}


/*
 * Phase 8: update IArray, IJMat, Wij, update EPSC
 */
__global__ void update_kernel_phase_8_gpu(
	int n,
	const void *ptr_tmp2, int w_tmp2,
	const void *ptr_hcu, int w_hcu,
//	const void *ptr_hcu_isp, int w_hcu_isp,
	const void *ptr_proj, int w_proj,
	const void *ptr_mcu, int w_mcu,
	const void *ptr_j_array, int w_j_array,
	const void *ptr_epsc, int w_epsc,
	void *ptr_i_array, int w_i_array,
	void *ptr_ij_mat, int w_ij_mat,
	void *ptr_wij, int w_wij,
	float timestamp, float prn){
	
	int idx;
	CUDA_KERNEL_LOOP(idx, n) {
		const int *ptr_tmp2_data = static_cast<const int *>(ptr_tmp2+idx*w_tmp2);
		int idx_i_array = ptr_tmp2_data[Database::IDX_TMP2_CONN];
		int idx_ij_mat = ptr_tmp2_data[Database::IDX_TMP2_IJ_MAT_INDEX];
		int idx_wij = idx_ij_mat;
		int idx_hcu = ptr_tmp2_data[Database::IDX_TMP2_DEST_HCU];
		int idx_subproj = ptr_tmp2_data[Database::IDX_TMP2_SUBPROJ];
		int idx_proj = ptr_tmp2_data[Database::IDX_TMP2_PROJ];
	//	int idx_hcu_isp = static_cast<const int *>(ptr_hcu+idx_hcu*w_hcu)[Database::IDX_HCU_ISP_INDEX]+idx_subproj;
	//	int idx_proj = static_cast<const int *>(ptr_hcu_isp+idx_hcu_isp*w_hcu_isp)[Database::IDX_HCU_ISP_VALUE];
	
		const float *ptr_proj_data = static_cast<const float *>(ptr_proj+idx_proj*w_proj);
		float kp = ptr_proj_data[Database::IDX_PROJ_TAUPDT]*prn;
		float ke = ptr_proj_data[Database::IDX_PROJ_TAUEDT];
		float kzi = ptr_proj_data[Database::IDX_PROJ_TAUZIDT];
		float kzj = ptr_proj_data[Database::IDX_PROJ_TAUZJDT];
		float wgain = ptr_proj_data[Database::IDX_PROJ_WGAIN];
		float eps = ptr_proj_data[Database::IDX_PROJ_EPS];
		float eps2 = ptr_proj_data[Database::IDX_PROJ_EPS2];
		float kfti = ptr_proj_data[Database::IDX_PROJ_KFTI];
		float prntaupdt = kp;
	
		// Update i_array
		float* ptr_i_array_data = static_cast<float *>(ptr_i_array+idx_i_array*w_i_array);
		float pi = ptr_i_array_data[Database::IDX_I_ARRAY_PI];
		float ei = ptr_i_array_data[Database::IDX_I_ARRAY_EI];
		float zi = ptr_i_array_data[Database::IDX_I_ARRAY_ZI];
		float ti = ptr_i_array_data[Database::IDX_I_ARRAY_TI];
		float pdt = timestamp - ti;
		if(pdt>0){
			pi = (pi - ((ei*kp*kzi - ei*ke*kp + ke*kp*zi)/(ke - kp) +
		                  (ke*kp*zi)/(kp - kzi))/(ke - kzi))/exp(kp*pdt) +
		      ((exp(kp*pdt - ke*pdt)*(ei*kp*kzi - ei*ke*kp + ke*kp*zi))/(ke - kp) +
		       (ke*kp*zi*exp(kp*pdt - kzi*pdt))/(kp - kzi))/(exp(kp*pdt)*(ke - kzi));
			ei = (ei - (ke*zi)/(ke - kzi))/exp(ke*pdt) +
		      (ke*zi*exp(ke*pdt - kzi*pdt))/(exp(ke*pdt)*(ke - kzi));
			zi = zi*exp(-kzi*pdt) + kfti;
			ti = timestamp;
		
			ptr_i_array_data[Database::IDX_I_ARRAY_PI]=pi;
			ptr_i_array_data[Database::IDX_I_ARRAY_EI]=ei;
			ptr_i_array_data[Database::IDX_I_ARRAY_ZI]=zi;
			ptr_i_array_data[Database::IDX_I_ARRAY_TI]=ti;
		}
	
	

		int mcu_num = static_cast<const int *>(ptr_hcu+idx_hcu*w_hcu)[Database::IDX_HCU_MCU_NUM];
		for(int i=0; i<mcu_num; i++){
			//Update ij_mat
			float* ptr_ij_mat_data = static_cast<float *>(ptr_ij_mat+(idx_ij_mat+i)*w_ij_mat);
			float pij = ptr_ij_mat_data[Database::IDX_IJ_MAT_PIJ];
			float eij = ptr_ij_mat_data[Database::IDX_IJ_MAT_EIJ];
			float zi = ptr_ij_mat_data[Database::IDX_IJ_MAT_ZI2];
			float zj = ptr_ij_mat_data[Database::IDX_IJ_MAT_ZJ2];
			float tij = ptr_ij_mat_data[Database::IDX_IJ_MAT_TIJ];
			float pdt = timestamp - tij;
		
			if(pdt>0)
			{
				pij = (pij + ((eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj)/(ke - kp) -
				            (ke*kp*zi*zj)/(kzi - kp + kzj))/(kzi - ke + kzj))/exp(kp*pdt) -
				    ((exp(kp*pdt - ke*pdt)*(eij*kp*kzi - eij*ke*kp + eij*kp*kzj + ke*kp*zi*zj))/(ke - kp) -
				     (ke*kp*zi*zj*exp(kp*pdt - kzi*pdt - kzj*pdt))/
				     (kzi - kp + kzj))/(exp(kp*pdt)*(kzi - ke + kzj));
				eij = (eij + (ke*zi*zj)/(kzi - ke + kzj))/exp(ke*pdt) -
				    (ke*zi*zj)/(exp(kzi*pdt)*exp(kzj*pdt)*(kzi - ke + kzj));
				zi = zi*exp(-kzi*pdt) + kfti;
				zj = zj*exp(-kzj*pdt);
				tij = timestamp;
			 	
			 	ptr_ij_mat_data[Database::IDX_IJ_MAT_PIJ]=pij;
				ptr_ij_mat_data[Database::IDX_IJ_MAT_EIJ]=eij;
				ptr_ij_mat_data[Database::IDX_IJ_MAT_ZI2]=zi;
				ptr_ij_mat_data[Database::IDX_IJ_MAT_ZJ2]=zj;
				ptr_ij_mat_data[Database::IDX_IJ_MAT_TIJ]=tij;
			}
	
			//Update wij and epsc
			if(prntaupdt){
				int idx_mcu = static_cast<const int *>(ptr_hcu+idx_hcu*w_hcu)[Database::IDX_HCU_MCU_INDEX]+i;
				int idx_j_array = static_cast<const int *>(ptr_mcu+idx_mcu*w_mcu)[Database::IDX_MCU_J_ARRAY_INDEX]+idx_subproj;
				float pj = static_cast<const float *>(ptr_j_array+(idx_j_array)*w_j_array)[Database::IDX_J_ARRAY_PJ];		
				float wij = wgain * log((pij + eps2)/((pi + eps)*(pj + eps)));
				float* ptr_wij_data = static_cast<float *>(ptr_wij+(idx_wij+i)*w_wij);
				ptr_wij_data[Database::IDX_WIJ_VALUE] = wij;
				float *ptr_epsc_data = static_cast<float *>(ptr_epsc+(idx_j_array)*w_epsc);
				atomicAdd(&ptr_epsc_data[IDX_EPSC_VALUE], wij);
			}
		}
	}
}

void Net::update_phase_8_gpu(){
	int h_tmp2=_tmp2->height();
	if(h_tmp2<=0)
		return;
	const void *ptr_tmp2 = _tmp2->gpu_data();
	int w_tmp2 = _tmp2->width();
	void *ptr_wij = _wij -> mutable_gpu_data();
	int w_wij = _wij->width();
	void *ptr_i_array = _i_array -> mutable_gpu_data();
	int w_i_array = _i_array->width();
	const void *ptr_hcu = _hcu -> gpu_data();
	int w_hcu = _hcu->width();
//	const void *ptr_hcu_isp = _hcu_isp -> gpu_data();
//	int w_hcu_isp = _hcu_isp->width();
	const void *ptr_proj = _proj -> gpu_data();
	int w_proj = _proj->width();
	const void *ptr_mcu = _mcu -> gpu_data();
	int w_mcu = _mcu->width();
	const void *ptr_j_array = _j_array -> gpu_data();
	int w_j_array = _j_array->width();
	const void *ptr_epsc = _epsc->gpu_data();
	int w_epsc = _epsc->width();
	void *ptr_ij_mat = _ij_mat -> mutable_gpu_data();
	int w_ij_mat = _ij_mat->width();
	
	const float *ptr_conf = static_cast<const float*>(_conf->cpu_data());
	float timestamp=ptr_conf[Database::IDX_CONF_TIMESTAMP];
	float prn=ptr_conf[Database::IDX_CONF_PRN];
	
	update_kernel_phase_8_gpu<<<GSBN_GET_BLOCKS(h_tmp2), GSBN_CUDA_NUM_THREADS>>>(
		h_tmp2,
		ptr_tmp2, w_tmp2,
		ptr_hcu, w_hcu,
//		ptr_hcu_isp, w_hcu_isp,
		ptr_proj, w_proj,
		ptr_mcu, w_mcu,
		ptr_j_array, w_j_array,
		ptr_epsc, w_epsc,
		ptr_i_array, w_i_array,
		ptr_ij_mat, w_ij_mat,
		ptr_wij, w_wij,
		timestamp, prn);
	CUDA_POST_KERNEL_CHECK;

/*	
	for(int i=0; i<h_tmp2;i++){
		const int *ptr_tmp2_data = static_cast<const int *>(_tmp2->cpu_data(i));
		int hcu_idx = ptr_tmp2_data[Database::IDX_TMP2_DEST_HCU];
		int sub_proj = ptr_tmp2_data[Database::IDX_TMP2_SUBPROJ];
		int wij_idx = ptr_tmp2_data[Database::IDX_TMP2_IJ_MAT_INDEX];
		const int *ptr_hcu=static_cast<const int*>(_hcu->cpu_data(hcu_idx));
		int mcu_idx = ptr_hcu[Database::IDX_HCU_MCU_INDEX];
		int mcu_num = ptr_hcu[Database::IDX_HCU_MCU_NUM];
		for(int j=0; j<mcu_num; j++){
			int epsc_idx = static_cast<const int *>(_mcu->cpu_data(mcu_idx+j))[Database::IDX_MCU_J_ARRAY_INDEX];
			float *ptr_epsc_data = static_cast<float *>(_epsc->mutable_cpu_data(epsc_idx+sub_proj));
			const float *ptr_wij_data = static_cast<const float *>(_wij->cpu_data(wij_idx+j));
			ptr_epsc_data[Database::IDX_EPSC_VALUE] += ptr_wij_data[Database::IDX_WIJ_VALUE];
		}
	}
*/
}

/*
 * Phase 9: deal with special spikes (REQ and ACK)
 */
void Net::update_phase_9_gpu(){
	_tmp3->reset();
	
	int h_conn0=_conn0->height();
	for(int i=0; i<h_conn0; i++){
		int *ptr_conn0 = static_cast<int *>(_conn0->mutable_cpu_data(i, 0));
		int queue=ptr_conn0[Database::IDX_CONN0_QUEUE];
		
		if(queue & 0x01){
			int *ptr_hcu_slot;
			int *ptr_conn;
			int *ptr_hcu;
			int idx_mcu_fanout;
			int *ptr_mcu_fanout;
			int idx_hcu;
			int mcu_num;
			void *ptr_tmp3;
			int* ptr_tmp30;
			float* ptr_tmp31;
			int proj_idx;
			int proj_mcu_num;
			float pi0;
			vector<int> *vec;
			vector<int>::iterator position;
			switch(ptr_conn0[Database::IDX_CONN0_TYPE]){
			case 1:	// REQ INCOMMING SPIKE
//				LOG(INFO) << ptr_conn0[Database::IDX_CONN0_DEST_HCU];
				ptr_hcu_slot = static_cast<int *>(_hcu_slot->mutable_cpu_data(ptr_conn0[Database::IDX_CONN0_DEST_HCU]));
				if(ptr_hcu_slot[Database::IDX_HCU_SLOT_VALUE]>0){
					ptr_hcu_slot[Database::IDX_HCU_SLOT_VALUE]--;
					ptr_conn0[Database::IDX_CONN0_TYPE]=2;
				}else{
					ptr_conn0[Database::IDX_CONN0_TYPE]=3;
				}
				queue |= (0x01 << (ptr_conn0[Database::IDX_CONN0_DELAY]));
				break;
			case 2:	// ACK INCOMMING SPIKE, ESTABLISH CONNECTION
				ptr_hcu = static_cast<int *>(_hcu->mutable_cpu_data(ptr_conn0[Database::IDX_CONN0_DEST_HCU]));
				mcu_num = ptr_hcu[Database::IDX_HCU_MCU_NUM];
				
				proj_idx = ptr_conn0[Database::IDX_CONN0_PROJ];
				pi0 = static_cast<const float *>(_proj->cpu_data(proj_idx))[Database::IDX_PROJ_PI0];
				proj_mcu_num = static_cast<const int *>(_proj->cpu_data(proj_idx))[Database::IDX_PROJ_MCU_NUM];
				// use tmp3 to initialize new connection
				ptr_tmp3 = _tmp3->expand(1);
				ptr_tmp30 = static_cast<int *>(ptr_tmp3);
				ptr_tmp30[Database::IDX_TMP3_CONN] = _conn->height();
				ptr_tmp30[Database::IDX_TMP3_DEST_HCU] = ptr_conn0[Database::IDX_CONN0_DEST_HCU];
				ptr_tmp30[Database::IDX_TMP3_IJ_MAT_IDX] = _ij_mat->height();
				ptr_tmp31 = static_cast<float *>(ptr_tmp3);
				ptr_tmp31[Database::IDX_TMP3_PI_INIT] = pi0;
				ptr_tmp31[Database::IDX_TMP3_PIJ_INIT] = 1.0/proj_mcu_num;
			
				ptr_conn = static_cast<int *>(_conn->expand(1));
				ptr_conn[Database::IDX_CONN_SRC_MCU] = ptr_conn0[Database::IDX_CONN0_SRC_MCU];
				ptr_conn[Database::IDX_CONN_DEST_HCU] = ptr_conn0[Database::IDX_CONN0_DEST_HCU];
				ptr_conn[Database::IDX_CONN_SUBPROJ] = ptr_conn0[Database::IDX_CONN0_SUBPROJ];
				ptr_conn[Database::IDX_CONN_PROJ] = ptr_conn0[Database::IDX_CONN0_PROJ];
				ptr_conn[Database::IDX_CONN_DELAY] = ptr_conn0[Database::IDX_CONN0_DELAY];
				ptr_conn[Database::IDX_CONN_QUEUE] = 0;
				ptr_conn[Database::IDX_CONN_IJ_MAT_INDEX] = _ij_mat->height();
				MemBlock::type_t t;
				_i_array->expand(1, &t);
				_ij_mat->expand(mcu_num, &t);
				_wij->expand(mcu_num, &t);
				ptr_conn0[Database::IDX_CONN0_TYPE] = 0;
				
				_empty_conn0_list.push_back(i);
				break;
			case 3:	// ACK INCOMMING SPIKE, REFUSE CONNECTION
				ptr_conn0[Database::IDX_CONN0_TYPE] = 0;	//set conn type to EMPTY, connection removed.
				idx_mcu_fanout = ptr_conn0[Database::IDX_CONN0_SRC_MCU];
				idx_hcu = ptr_conn0[Database::IDX_CONN0_DEST_HCU];
				ptr_mcu_fanout = static_cast<int*>(_mcu_fanout->mutable_cpu_data(idx_mcu_fanout));
				ptr_mcu_fanout[Database::IDX_MCU_FANOUT_VALUE]++;	// Recovery the fanout
				// Update the empty row list. It will be reused to establish new connections.
				_empty_conn0_list.push_back(i);
				vec = &(_existed_conn_list[idx_mcu_fanout]);
				position = find(vec->begin(), vec->end(), idx_hcu);
				if (position != vec->end())
					vec->erase(position);
				break;
			default:
				break;
			}
		}
		ptr_conn0[Database::IDX_CONN0_QUEUE] = queue >> 1;
	}
}


/*
 * Phase 10: initialize new IArray, IJMat
 */

__global__ void update_kernel_phase_10_gpu(
	int n,
	const void *ptr_tmp3, int w_tmp3,
	const void *ptr_hcu, int w_hcu,
	void *ptr_i_array, int w_i_array,
	void *ptr_ij_mat, int w_ij_mat){
	
	int idx;
	CUDA_KERNEL_LOOP(idx, n) {
		const int *ptr_tmp3_data=static_cast<const int*>(ptr_tmp3+idx*w_tmp3);
		int i_array_idx=ptr_tmp3_data[Database::IDX_TMP3_CONN];
		int hcu_idx=ptr_tmp3_data[Database::IDX_TMP3_DEST_HCU];
		int ij_mat_idx = ptr_tmp3_data[Database::IDX_TMP3_IJ_MAT_IDX];
		const float *ptr_tmp3_data0=static_cast<const float*>(ptr_tmp3+idx*w_tmp3);
		float pi_init = ptr_tmp3_data0[Database::IDX_TMP3_PI_INIT];
		float pij_init = ptr_tmp3_data0[Database::IDX_TMP3_PIJ_INIT];
		static_cast<float *>(ptr_i_array+i_array_idx*w_i_array)[Database::IDX_I_ARRAY_PI] = pi_init;
		int mcu_num = static_cast<const int*>(ptr_hcu+hcu_idx*w_hcu)[Database::IDX_HCU_MCU_NUM];
		for(int i=0; i<mcu_num; i++){
			static_cast<float *>(ptr_ij_mat+(ij_mat_idx+i)*w_ij_mat)[Database::IDX_IJ_MAT_PIJ] = pij_init;
		}
	}
}

void Net::update_phase_10_gpu(){
	int h_tmp3 = _tmp3->height();
	if(h_tmp3<=0){
		return;
	}
	const void *ptr_tmp3 = _tmp3->gpu_data();
	int w_tmp3 = _tmp3->width();
	const void *ptr_hcu = _hcu->gpu_data();
	int w_hcu = _hcu->width();
	void *ptr_i_array = _i_array->mutable_gpu_data();
	int w_i_array = _i_array->width();
	void *ptr_ij_mat = _ij_mat->mutable_gpu_data();
	int w_ij_mat = _ij_mat->width();
	
	update_kernel_phase_10_gpu<<<GSBN_GET_BLOCKS(h_tmp3), GSBN_CUDA_NUM_THREADS>>>(
		h_tmp3,
		ptr_tmp3, w_tmp3,
		ptr_hcu, w_hcu,
		ptr_i_array, w_i_array,
		ptr_ij_mat, w_ij_mat);
	CUDA_POST_KERNEL_CHECK;
}


/*
 * Phase 11: Send special spikes
 */
void Net::update_phase_11_gpu(){
	int h_tmp1 = _tmp1->height();
	for(int i=0; i<h_tmp1; i++){
		int idx_mcu = static_cast<const int *>(_tmp1->cpu_data(i))[Database::IDX_TMP1_MCU_IDX];
		int *ptr_mcu_fanout = static_cast<int *>(_mcu_fanout->mutable_cpu_data(idx_mcu));
		if(*ptr_mcu_fanout>0){
			*ptr_mcu_fanout--;
			
			const int *ptr_addr = static_cast<const int *>(_addr->cpu_data(idx_mcu));
			int idx_pop=ptr_addr[Database::IDX_ADDR_POP];
			int idx_hcu=ptr_addr[Database::IDX_ADDR_HCU];

			const int *ptr_hcu = static_cast<const int *>(_hcu->cpu_data(idx_hcu));
			int idx_hcuproj=ptr_hcu[Database::IDX_HCU_OSP_INDEX];
			int num_hcuproj=ptr_hcu[Database::IDX_HCU_OSP_NUM];
			vector<int> proj_list;
			for(int j=0; j<num_hcuproj; j++){
				int proj_val = static_cast<const int *>(_hcu_osp->cpu_data(idx_hcuproj+j))[Database::IDX_HCU_OSP_VALUE];
				proj_list.push_back(proj_val);
			}
			vector<int> list_available_hcu;
			vector<int> list_available_proj;
			for(vector<int>::iterator it=proj_list.begin(); it!=proj_list.end(); it++){
				int dest_pop = static_cast<const int *>(_proj->cpu_data(*it))[Database::IDX_PROJ_DEST_POP];
				int iii_hcu = static_cast<const int *>(_pop->cpu_data(dest_pop))[Database::IDX_POP_HCU_INDEX];
				int nnn_hcu = static_cast<const int *>(_pop->cpu_data(dest_pop))[Database::IDX_POP_HCU_NUM];
				for(int k=0;k<nnn_hcu;k++){
					list_available_hcu.push_back(iii_hcu+k);
					list_available_proj.push_back(*it);
				}
			}
			vector<int> list=_existed_conn_list[idx_mcu];
			for(vector<int>::iterator it=list.begin(); it!=list.end();it++){
				vector<int>::iterator position = find(list_available_hcu.begin(), list_available_hcu.end(), *it);
				if (position != list_available_hcu.end()){
					list_available_hcu.erase(position);
					list_available_proj.erase(list_available_proj.begin()+distance(list_available_hcu.begin(), position));
				}
			}
			if(list_available_hcu.size()<=0){
				*ptr_mcu_fanout++;
				continue;
			}
			int idx_target_hcu = ceil(Random::gen_uniform01()*list_available_hcu.size()-1);
			int target_hcu = list_available_hcu[idx_target_hcu];
			int target_proj = list_available_proj[idx_target_hcu];
			int target_subproj = 0;
			int target_hcu_isp_idx = static_cast<const int*>(_hcu->cpu_data(target_hcu))[Database::IDX_HCU_ISP_INDEX];
			int target_hcu_isp_num = static_cast<const int*>(_hcu->cpu_data(target_hcu))[Database::IDX_HCU_ISP_NUM];
			for(int l=0;l<target_hcu_isp_num;l++){
				if(static_cast<const int*>(_hcu_isp->cpu_data(target_hcu_isp_idx+l))[Database::IDX_HCU_ISP_VALUE] == target_proj){
					target_subproj=l;
					break;
				}
			}
			
			int *ptr_conn0;
			if(_empty_conn0_list.empty()){
				ptr_conn0 = static_cast<int*>(_conn0->expand(1));
			}else{
				int index = _empty_conn0_list.back();
				_empty_conn0_list.pop_back();
				ptr_conn0 = static_cast<int*>(_conn0->mutable_cpu_data(index));
			}
			
			ptr_conn0[Database::IDX_CONN0_SRC_MCU] = idx_mcu;
			ptr_conn0[Database::IDX_CONN0_DEST_HCU] = target_hcu;
			ptr_conn0[Database::IDX_CONN0_SUBPROJ] = target_subproj;
			ptr_conn0[Database::IDX_CONN0_PROJ] = target_proj;
			ptr_conn0[Database::IDX_CONN0_DELAY] = __DELAY__;	// FIXME
			ptr_conn0[Database::IDX_CONN0_QUEUE] = 1 << __DELAY__-1; // FIXME
			ptr_conn0[Database::IDX_CONN0_TYPE] = 1;
			
			_existed_conn_list[idx_mcu].push_back(target_hcu);
		}
	}
}


/*
void Net::create_kernel_cpu(){
	unsigned char spk = static_cast<unsigned char *>(ptr_spk+index*w_spike);
	if(spk!=0){
		int fanout = static_cast<int *>(ptr_mcu_fanout+index*w_mcu_fanout)[0];
		if(fanout>0){
			static_cast<int *>(ptr_mcu_fanout+index*w_mcu_fanout)[0]=fanout-1;
			vector<int> list=conn_list[index];
			
			int mcu_pop = static_cast<int *>(ptr_addr+index*w_addr)[0];
			
			for(int i=0; i<h_proj; i++){
				if(static_cast<int *>(ptr_proj+i*w_proj)[0]==mcu_pop){
					hcu_pop=static_cast<int *>(ptr_proj+i*w_proj)[1];
					int first_hcu = static_cast<int *>(ptr_pop+i*w_pop)[0];
					int num_hcu = static_cast<int *>(ptr_pop+i*w_pop)[1];
					list2 = Random::select(num_hcu, num_hcu);
					for(int j=0; j<list2.size(); j++){
						int tmp = list2[j]+first_hcu;
						for(int k=0; k<list.size(); k++){
							if(tmp==list[k]){
								list2.erase(k);
								break;
							}
						}
						tmp_idx=ceil((Random::gen_uniform01()*list2.size()));
						hcu_idx=list2[tmp_idx];
						list.push_back(hcu_idx);
						
						int *ptr_conn;
						int idx;
						if(!vacum_list.empy()){
							idx=vacum_list.pop();
							ptr_conn = static_cast<int*>(_conn.mutable_cpu_data(idx));
						}else{
							idx=_conn.rows();
							ptr_conn = static_cast<int*>(_conn.append(1));
							_ij_mat.append_gpu(1);
						}
						ptr_conn[0]=index;
						ptr_conn[1]=hcu_idx;
						ptr_conn[2]=1;
						ptr_conn[3]=0x01;
						
						int *ptr_sel=static_cast<int*>(append(1));
						ptr_sel[0]=idx;
						
						return;
					}
				}
			}
		}
	}
}

void Net::create_cpu(){
	r=_spike.rows();
	for(int i=0; i<r; i++){
		create_kernel_cpu(void *ptr, int index, int w);
	}
}
*/

/*
 * Phase 12: increase Zj2
 */
__global__ void update_kernel_phase_12_gpu(
	int n,
	const void *ptr_tmp2, int w_tmp2,
	const void *ptr_proj, int w_proj,
	void *ptr_ij_mat, int w_ij_mat){
	
	int idx;
	CUDA_KERNEL_LOOP(idx, n) {
		const int *ptr_tmp2_data = static_cast<const int *>(ptr_tmp2+idx*w_tmp2);
		int idx_ij_mat = ptr_tmp2_data[Database::IDX_TMP2_IJ_MAT_INDEX];
		int idx_proj = ptr_tmp2_data[Database::IDX_TMP2_PROJ];
		const float *ptr_proj_data = static_cast<const float *>(ptr_proj+idx_proj*w_proj);
		float kftj = ptr_proj_data[Database::IDX_PROJ_KFTJ];
	
		int *ptr_ij_mat_data = static_cast<int *>(ptr_ij_mat+idx_ij_mat*w_ij_mat);
		ptr_ij_mat_data[Database::IDX_IJ_MAT_ZJ2] += kftj;
	}
}

void Net::update_phase_12_gpu(){
	_tmp2->reset();
	int h_tmp1=_tmp1->height();
	for(int i=0; i<h_tmp1; i++){
		int mcu=static_cast<const int *>(_tmp1->cpu_data(i))[Database::IDX_TMP1_MCU_IDX];
		int hcu=static_cast<const int *>(_addr->cpu_data(mcu))[Database::IDX_ADDR_HCU];
		int h_conn=_conn->height();
		for(int j=0; j<h_conn; j++){
			const int *ptr_conn = static_cast<const int *>(_conn->cpu_data(j));
			int src_mcu=ptr_conn[Database::IDX_CONN_SRC_MCU];
			int dest_hcu=ptr_conn[Database::IDX_CONN_DEST_HCU];
			if(dest_hcu==hcu){
				int ij_mat_first=static_cast<const int *>(_conn->cpu_data(j))[Database::IDX_CONN_IJ_MAT_INDEX];
				int offset = mcu - static_cast<const int *>(_hcu->cpu_data(hcu))[Database::IDX_HCU_MCU_INDEX];
				int ij_mat_idx = ij_mat_first + offset;
				int *ptr = static_cast<int*>(_tmp2->expand(1));
				ptr[Database::IDX_TMP2_IJ_MAT_INDEX]=ij_mat_idx;
				ptr[Database::IDX_TMP2_PROJ]=static_cast<const int *>(_conn->cpu_data(j))[Database::IDX_CONN_PROJ];
			}
			if(src_mcu==mcu){
				int delay = ptr_conn[Database::IDX_CONN_DELAY];
				static_cast<int *>(_conn->mutable_cpu_data(j))[Database::IDX_CONN_QUEUE] |= (1 << (delay-1));
			}
		}
	}
	int h_tmp2=_tmp2->height();
	if(h_tmp2<=0)
		return;
	const void *ptr_tmp2 = _tmp2->gpu_data();
	int w_tmp2 = _tmp2->width();
	const void *ptr_proj = _proj->gpu_data();
	int w_proj = _proj->width();
	void *ptr_ij_mat = _ij_mat->mutable_gpu_data();
	int w_ij_mat = _ij_mat->width();
	
	update_kernel_phase_12_gpu<<<GSBN_GET_BLOCKS(h_tmp2), GSBN_CUDA_NUM_THREADS>>>(
		h_tmp2,
		ptr_tmp2, w_tmp2,
		ptr_proj, w_proj,
		ptr_ij_mat, w_ij_mat);
	CUDA_POST_KERNEL_CHECK;
}
}
