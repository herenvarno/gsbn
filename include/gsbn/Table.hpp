#ifndef __GSBN_TABLE_HPP__
#define __GSBN_TABLE_HPP__

#include "gsbn/Common.hpp"
#include "gsbn/MemBlock.hpp"

namespace gsbn{

#define TABLE_DESC_INDEX_SIZE       0
#define TABLE_DESC_INDEX_HEIGHT     1
#define TABLE_DESC_INDEX_MAXHEIGHT  2
#define TABLE_DESC_INDEX_BLKHEIGHT  3
#define TABLE_DESC_INDEX_WIDTH      4
#define TABLE_DESC_INDEX_COLUMNS    5

/**
 * \class Table
 * \bref The class Table organize its data like a table.
 *
 * The data held by Table class can keep inforamtion in both CPU and GPU memory.
 * The information are automatically sychronized thanks to the mechanism of
 * MemBlock class. Please note that, the description of the data memory block is
 * also stored in memory as a MemBlock object, but it should be used only by CPU,
 * which means the information won't be sychronized to GPU memory.
 */
class Table{

public:
	
	/**
	 * \fn Table()
	 * \bref The constructor of the class Table. The table shouldn't be used until
	 * the init() function is called.
	 */
	Table();
	
	/**
	 * \fn Table(vector<int> fields, int block_height=10)
	 * \bref The constructor of the class Table. It also initializes the table.
	 * \param name The name of the table.
	 * \param fields The size (Byte) of each column.
	 * \param block_height The number of rows to increase while expanding the
	 * table.
	 */
	explicit Table(string name, vector<int> fields, int block_height=100);
	
	/**
	 * \fn init()
	 * \bref Initialize the table. define the shape of the table.
	 * \param name The name of the table.
	 * \param fields The size (Byte) of each column.
	 * \param block_height The number of rows to increase while expanding the
	 * table.
	 */
	void init(string name, vector<int> fields, int block_height=10);
	
	/**
	 * \fn expand()
	 * \bref Expand the table. Allocate larger new memory block.
	 * \param rows The number of row to be added.
	 * \param block_type The pointer which points to the container of output block
	 * type.
	 * \return The pointer to the entry of expanded part of the table.
	 *
	 * The expand() function can perform CPU memory expansion or GPU memory
	 * depending on the memory block type. If the output type container pointer
	 * \p block_type is NULL, the function will perform a CPU expansion. Otherwise,
	 * the type of expansion depends on the type of memory block before expansion.
	 * The return pointer type is also depends on the expansion type.
	 *
	 * If you want a CPU expansion, just set \p block_type as NULL. If you want a
	 * GPU expansion, synchronize the data to GPU memory before expand() is called.
	 * You can use gpu_data() function to achieve this. And don't forget pass a
	 * non-NULL pointer to \p block_type. After GPU expansion, it's better to
	 * check the block_type to make sure the memory block is now
	 * MemBlock::GPU_MEM_BLOCK before using the returned pointer.
	 */
	void* expand(int rows, MemBlock::type_t* block_type=NULL);
	
	const string name(){return _name;}
	/**
	 * \fn rows()
	 * \bref Get rows
	 * \return The number of rows.
	 */
	const int rows();
	/**
	 * \fn cols()
	 * \bref Get columns
	 * \return The number of columns.
	 */
	const int cols();
	/**
	 * \fn height()
	 * \bref Get height
	 * \return The height.
	 */
	const int height();
	/**
	 * \fn width()
	 * \bref Get width.
	 * \return The width.
	 */
	const int width();
	/**
	 * \fn field()
	 * \bref Get the size of one filed
	 * \param index The index of the filed.
	 * \return The size.
	 */
	const int field(int index);
	/**
	 * \fn offset()
	 * \bref Get the offset of a coordinate.
	 * \param row The row.
	 * \param col The column.
	 * \return The offset.
	 */
	const int offset(int row=0, int col=0);
	
	/**
	 * \fn cpu_data()
	 * \bref Get pointer of the CPU memory block. The information is garanteed to
	 * be up-to-date. The memory is read-only.
	 * \param row The row of the table.
	 * \param col The column of the table.
	 * \return The pointer to the memory.
	 */
	const void *cpu_data(int row=0, int col=0);
	/**
	 * \fn gpu_data()
	 * \bref Get pointer of the GPU memory block. The information is garanteed to
	 * be up-to-date. The memory is read-only.
	 * \param row The row of the table.
	 * \param col The column of the table.
	 * \return The pointer to the memory.
	 */
	const void *gpu_data(int row=0, int col=0);
	/**
	 * \fn mutable_cpu_data()
	 * \bref Get pointer of the CPU memory block. The information is garanteed to
	 * be up-to-date.
	 * \param row The row of the table.
	 * \param col The column of the table.
	 * \return The pointer to the memory.
	 */
	void *mutable_cpu_data(int row=0, int col=0);
	/**
	 * \fn mutable_gpu_data()
	 * \bref Get pointer of the GPU memory block. The information is garanteed to
	 * be up-to-date.
	 * \param row The row of the table.
	 * \param col The column of the table.
	 * \return The pointer to the memory.
	 */
	void *mutable_gpu_data(int row=0, int col=0);
	
	/**
	 * \fn dump()
	 * \bref Dump the memory data to a string.
	 * \return The string.
	 */
	virtual const string dump();
	
	TableState state();
	
	void set_state(TableState tab_st);
	
	void reset();
	
private:
	const int get_desc_item_cpu(int index);
	void set_desc_item_cpu(int index, int value);
	void set_name(string name);
	void set_block_height(int block_height);
	void set_fields(vector<int> fields);
	bool lock();
	bool expand_core(MemBlock::type_t block_type);
	MemBlock::type_t type();

	bool _locked;
	string _name;
	shared_ptr<MemBlock> _desc;
	shared_ptr<MemBlock> _data;

};

}
#endif //__GSBN_TABLE_HPP__
