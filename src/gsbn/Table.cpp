#include "gsbn/Table.hpp"

namespace gsbn{

Table::Table(): _name(), _locked(false), _desc(new MemBlock()) ,_data(new MemBlock()) {

}

Table::Table(string name, vector<int> fields, int block_height) : _locked(false), _desc(), _data(){
	init(name, fields, block_height);
}

void Table::init(string name, vector<int> fields, int block_height){
	CHECK(!_locked)
		<< "Multiple table init function calls!";
	
	CHECK_GT(fields.size(), 0);
	set_name(name);
	set_fields(fields);
	set_block_height(block_height);
	lock();
}

void* Table::expand(int rows, MemBlock::type_t* block_type){
	CHECK(_locked)
		<< "Table should be locked after initialization."
		<< "Only locked tables are allowed to be filled with data!";

	CHECK_GE(rows, 0);
	
	if(rows==0){
		LOG(WARNING) << "Table append 0 row. pass!";
		return NULL;
	}
	
	int max_height = get_desc_item_cpu(TABLE_DESC_INDEX_MAXHEIGHT);
	int blk_height = get_desc_item_cpu(TABLE_DESC_INDEX_BLKHEIGHT);
	int height = get_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT);
	int width = get_desc_item_cpu(TABLE_DESC_INDEX_WIDTH);
	MemBlock::type_t t=type();
	if(!block_type){
		t=MemBlock::CPU_MEM_BLOCK;
	}
	while(rows>max_height-height){
		expand_core(t);
		max_height = get_desc_item_cpu(TABLE_DESC_INDEX_MAXHEIGHT);
	}
	
	set_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT, height+rows);
	void* ptr=NULL;
	if(block_type){
		*block_type=type();
		if(*block_type==MemBlock::GPU_MEM_BLOCK){
			ptr=_data->mutable_gpu_data()+offset(height, 0);
		}else{
			ptr=_data->mutable_cpu_data()+offset(height, 0);
		}
	}else{
		ptr=_data->mutable_cpu_data()+offset(height, 0);
	}
	return ptr; 
}

const int Table::rows(){
	return get_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT);
}
const int Table::cols() {
	return get_desc_item_cpu(TABLE_DESC_INDEX_SIZE)-TABLE_DESC_INDEX_COLUMNS;
}
const int Table::height() {
	return get_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT);
}
const int Table::width() {
	return get_desc_item_cpu(TABLE_DESC_INDEX_WIDTH);
}
const int Table::field(int index) {
	CHECK_GE(index, 0);
	return get_desc_item_cpu(TABLE_DESC_INDEX_COLUMNS+index);
}
const int Table::offset(int row, int col) {
	CHECK_GE(row, 0);
	CHECK_GE(col, 0);
	int maxcol = get_desc_item_cpu(TABLE_DESC_INDEX_SIZE)-TABLE_DESC_INDEX_COLUMNS;
	int maxrow = get_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT);
	CHECK_LT(row, maxrow);
	CHECK_LT(col, maxcol);
	
	int width = get_desc_item_cpu(TABLE_DESC_INDEX_WIDTH);
	int os= row * width;
	for(int i=0; i<col; i++){
		os += get_desc_item_cpu(TABLE_DESC_INDEX_COLUMNS+i);
	}
	return static_cast<const int>(os);
}

const void* Table::cpu_data(int row, int col){
	const int os = offset(row, col);
	return _data->cpu_data()+os;
	
}
const void* Table::gpu_data(int row, int col){
	__NOT_IMPLEMENTED__
}
void *Table::mutable_cpu_data(int row, int col){
	const int os = offset(row, col);
	return _data->mutable_cpu_data()+os;

}
void *Table::mutable_gpu_data(int row, int col){
	__NOT_IMPLEMENTED__
}


const string Table::dump(){
	CHECK(_locked)
		<< "Table should be locked after initialization."
		<< "Only locked tables are allowed to be filled with data!";
		
	int height = get_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT);
	int width = get_desc_item_cpu(TABLE_DESC_INDEX_WIDTH);
	const unsigned char *data_ptr;
	data_ptr=static_cast<const unsigned char*>(_data->cpu_data());	
	ostringstream s;
	for(int i=0; i<height; i++){
		for(int j=0; j<width; j++){
			s << hex << setfill('0') << setw(2) << static_cast<unsigned int>(data_ptr[i*width+j]) << " ";
		}
		s << endl;
	}
	std::string str =  s.str();
	return str;
}


TableState Table::state(){
	CHECK(_locked)
		<< "Table should be locked after initialization.";
	LOG(INFO) << "11";
	TableState tab_st;
	tab_st.set_name(_name);
	LOG(INFO) << "12";
	size_t data_size = height() * width();
	size_t desc_size = get_desc_item_cpu(TABLE_DESC_INDEX_SIZE) * sizeof(int);
	tab_st.set_desc(_desc->cpu_data(), desc_size);
	tab_st.set_data(_data->cpu_data(), data_size);
	
	return tab_st;
}

void Table::set_state(TableState tab_st){
	size_t desc_size = get_desc_item_cpu(TABLE_DESC_INDEX_SIZE) * sizeof(int);
	
	const string desc_str=tab_st.desc();
	const string data_str=tab_st.data();
	
	size_t desc_size_0 = desc_str.size();
	size_t data_size_0 = data_str.size();
	if(desc_size!=desc_size_0){
		
		LOG(FATAL) << "Table not match! Abort!" << desc_size << " " << desc_size_0;
	}
	const int *desc = static_cast<const int*>(_desc->cpu_data());
	const int *desc_0 = (const int *)desc_str.c_str();
	
	if(desc[TABLE_DESC_INDEX_SIZE]==desc_0[TABLE_DESC_INDEX_SIZE] &&
		desc[TABLE_DESC_INDEX_WIDTH]==desc_0[TABLE_DESC_INDEX_WIDTH] ){
			for(size_t i=TABLE_DESC_INDEX_COLUMNS; i<desc[TABLE_DESC_INDEX_SIZE]-TABLE_DESC_INDEX_COLUMNS; i++){
				if(desc[i] != desc_0[i]){
					LOG(FATAL) << "Table not match! Abort!";
				}
			}
	}else{
		LOG(FATAL) << "Table not match! Abort!" << int(desc[TABLE_DESC_INDEX_WIDTH]) << " " << int(desc_0[TABLE_DESC_INDEX_WIDTH]);
	}
	
	int r=data_size_0/width();
	set_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT, 0);
	expand(r);
	const char *data_0 = data_str.c_str();
	size_t data_size=data_str.size();
	void *data = mutable_cpu_data();
	MemBlock::memcpy_cpu_to_cpu(data, data_0, data_size);
}

/*
 * Private functions
 */

void Table::set_name(string name){
	CHECK(!name.empty());
	_name = name;
}

void Table::set_block_height(int block_height){
	CHECK_GT(block_height, 0);
	set_desc_item_cpu(TABLE_DESC_INDEX_BLKHEIGHT, block_height);
}

void Table::set_fields(vector<int> fields){
	CHECK_GT(fields.size(), 0);
	
	int size=TABLE_DESC_INDEX_COLUMNS+fields.size();
	
	
	shared_ptr<MemBlock> new_desc(new MemBlock(size*sizeof(int)));
	int *new_desc_ptr=static_cast<int *>(new_desc->mutable_cpu_data());
	const int *old_desc_ptr=static_cast<const int*>(_desc->cpu_data());
	
	if(old_desc_ptr && _desc->size()>TABLE_DESC_INDEX_COLUMNS*sizeof(int)){
		for(int i=0; i<TABLE_DESC_INDEX_COLUMNS; i++){
			new_desc_ptr[i] = old_desc_ptr[i];
		}
	}

	_desc=new_desc;
	set_desc_item_cpu(TABLE_DESC_INDEX_SIZE, TABLE_DESC_INDEX_COLUMNS+fields.size());
	
	int width=0;
	for(int i=0; i<fields.size(); i++){
		CHECK_GT(fields[i], 0);
		set_desc_item_cpu(TABLE_DESC_INDEX_COLUMNS+i, fields[i]);
		width += fields[i];
	}
	set_desc_item_cpu(TABLE_DESC_INDEX_WIDTH, width);
}

const int Table::get_desc_item_cpu(int index){
	CHECK_GE(index, 0);
	if(index>=TABLE_DESC_INDEX_COLUMNS){
		int size = static_cast<const int*>(_desc->cpu_data())[TABLE_DESC_INDEX_SIZE];
		CHECK_LT(index, size) << "Illegal field index!";
	}
	return static_cast<const int*>(_desc->mutable_cpu_data())[index];
}

void Table::set_desc_item_cpu(int index, int value){
	CHECK_GE(index, 0);
	if(index>=TABLE_DESC_INDEX_COLUMNS){
		int size = static_cast<const int*>(_desc->cpu_data())[TABLE_DESC_INDEX_SIZE];
		CHECK_LT(index, size);
	}
	static_cast<int*>(_desc->mutable_cpu_data())[index]=value;
}

bool Table::lock(){
	CHECK(!_name.empty());
	CHECK(_desc->cpu_data());
	
	int size = get_desc_item_cpu(TABLE_DESC_INDEX_SIZE);
	int height = get_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT);
	int max_height = get_desc_item_cpu(TABLE_DESC_INDEX_MAXHEIGHT);
	int blk_height = get_desc_item_cpu(TABLE_DESC_INDEX_BLKHEIGHT);
	int width = get_desc_item_cpu(TABLE_DESC_INDEX_WIDTH);
	
	CHECK_GT(size, TABLE_DESC_INDEX_COLUMNS);
	CHECK_GE(height, 0);
	CHECK_GE(max_height, height);
	CHECK_GT(blk_height, 0);
	CHECK_GT(width, 0);
	
	_locked=true;
	return _locked;
}

bool Table::expand_core(MemBlock::type_t block_type){

	int old_height = get_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT);
	int max_height = get_desc_item_cpu(TABLE_DESC_INDEX_MAXHEIGHT);
	int blk_height = get_desc_item_cpu(TABLE_DESC_INDEX_BLKHEIGHT);
	int width = get_desc_item_cpu(TABLE_DESC_INDEX_WIDTH);
	
	int new_height = max_height+blk_height;
	size_t new_mem_size = new_height*width*sizeof(char);
	size_t old_mem_size = old_height*width*sizeof(char);
	
	shared_ptr<MemBlock> new_data(new MemBlock(new_mem_size));
	if(block_type==MemBlock::GPU_MEM_BLOCK){
		void *new_data_ptr = new_data->mutable_gpu_data();
		const void* old_data_ptr = _data->mutable_gpu_data();
		if(old_data_ptr && _data->size()>0){
			MemBlock::memcpy_gpu_to_gpu(new_data_ptr, old_data_ptr, old_mem_size);
		}
	}else{
		void *new_data_ptr = new_data->mutable_cpu_data();
		const void* old_data_ptr = _data->mutable_cpu_data();
		if(old_data_ptr && _data->size()>0){
			MemBlock::memcpy_cpu_to_cpu(new_data_ptr, old_data_ptr, old_mem_size);
		}
	}
	
	_data=new_data;
	set_desc_item_cpu(TABLE_DESC_INDEX_MAXHEIGHT, new_height);
	LOG(INFO) << "EXPAND TO: "<<new_height;
 	return true;
}

MemBlock::type_t Table::type(){
	return _data->type();
}

}
