#include "gsbn/Table.hpp"

namespace gsbn{

Table::Table(): _locked(false), _desc(new MemBlock()) ,_data(new MemBlock()) {

}

Table::Table(vector<int> fields, int block_height) : _locked(false), _desc(), _data(){
	init(fields, block_height);
}

void Table::init(vector<int> fields, int block_height){
	CHECK(!_locked)
		<< "Multiple table init function calls!";
	
	CHECK_GT(fields.size(), 0);
	set_fields(fields);
	set_block_height(block_height);
	lock();
}

void* Table::expand(int rows){
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
	while(rows>max_height-height){
		expand_core();
		max_height = get_desc_item_cpu(TABLE_DESC_INDEX_MAXHEIGHT);
	}
	
	set_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT, height+rows);
	void* ptr=_data->mutable_cpu_data()+offset(height, 0);
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


/*
 * Private functions
 */

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

bool Table::expand_core(){

	int old_height = get_desc_item_cpu(TABLE_DESC_INDEX_HEIGHT);
	int max_height = get_desc_item_cpu(TABLE_DESC_INDEX_MAXHEIGHT);
	int blk_height = get_desc_item_cpu(TABLE_DESC_INDEX_BLKHEIGHT);
	int width = get_desc_item_cpu(TABLE_DESC_INDEX_WIDTH);
	
	int new_height = max_height+blk_height;
	size_t new_mem_size = new_height*width*sizeof(char);
	size_t old_mem_size = old_height*width*sizeof(char);
	
	
	shared_ptr<MemBlock> new_data(new MemBlock(new_mem_size));
	void *new_data_ptr = new_data->mutable_cpu_data();
	const void* old_data_ptr = _data->mutable_cpu_data();
	if(old_data_ptr && _data->size()>0){
		memcpy(new_data_ptr, old_data_ptr, old_mem_size);
	}
	_data=new_data;
	set_desc_item_cpu(TABLE_DESC_INDEX_MAXHEIGHT, new_height);
	LOG(INFO) << "EXPAND TO: "<<new_height;
 	return true;
}

}
