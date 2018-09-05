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
	void **value;
	void *old_value;
};

linked_list *create_linked_list(void *value);

linked_list *add_linked_list(linked_list *dest, linked_list *element);

CUDA_struct *create_CUDA_struct(void **value, size_t size);

void CUDA_struct_add_child(CUDA_struct *parent, CUDA_struct *child);

void *CUDA_struct_compile(CUDA_struct *parent);

void *CUDA_struct_free(CUDA_struct *parent);

