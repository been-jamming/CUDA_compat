#include <stdlib.h>

typedef struct linked_list linked_list;

struct linked_list{
	void *value;
	linked_list *next;
};

typedef struct CUDA_struct CUDA_struct;

struct CUDA_struct{
	linked_list *children;
	size_t size;
	void *value;
	void *old_value;
	unsigned char allocated;
	unsigned char freeing;
};

typedef struct CUDA_child CUDA_child;

struct CUDA_child{
	CUDA_struct *struct_var;
	void **pointer;
	void *old_pointer;
};

linked_list *create_linked_list(void *value);

linked_list *add_linked_list(linked_list *dest, linked_list *element);

CUDA_child *create_CUDA_child(CUDA_struct *struct_var, void **pointer);

CUDA_struct *create_CUDA_struct(void *value, size_t size);

void CUDA_struct_add_child(CUDA_struct *parent, CUDA_struct *child, void **pointer);

void CUDA_struct_compile(CUDA_struct *parent);

void *CUDA_struct_free(CUDA_struct *parent);

void *CUDA_struct_pull(CUDA_struct *parent);

void *CUDA_struct_push(CUDA_struct *parent);

