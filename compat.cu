#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include "compat.h"

int *t_ans;
int *t2_ans;

struct test{
	int x;
	struct test *child;
	struct test *parent;
};

linked_list *create_linked_list(void *value){
	linked_list *output;
	output = (linked_list *) malloc(sizeof(linked_list));
	output->next = (linked_list *) 0;
	output->value = value;
	return output;
}

linked_list *add_linked_list(linked_list *dest, linked_list *element){
	element->next = dest;
	return element;
}

CUDA_child *create_CUDA_child(CUDA_struct *struct_var, void **pointer){
	CUDA_child *output;
	output = (CUDA_child *) malloc(sizeof(CUDA_child));
	output->struct_var = struct_var;
	output->pointer = pointer;
	output->old_pointer = (void *) 0;
	return output;
}

CUDA_struct *create_CUDA_struct(void *pointer, size_t size){
	CUDA_struct *output;
	output = (CUDA_struct *) malloc(sizeof(CUDA_struct));
	output->children = (linked_list *) 0;
	output->value = (void *) 0;
	output->old_value = pointer;
	output->size = size;
	output->allocated = 0;
	output->freeing = 0;
	return output;
}

void CUDA_struct_add_child(CUDA_struct *parent, CUDA_struct *child, void **pointer){
	parent->children = add_linked_list(parent->children, create_linked_list((void *) create_CUDA_child(child, pointer)));
}

void CUDA_struct_compile(CUDA_struct *parent, void **pointer){
	linked_list *child;
	void **child_pointer;
	CUDA_struct *child_struct;
	
	if(parent->allocated){
		*pointer = parent->value;
		return;
	}

	cudaMalloc(&(parent->value), parent->size);
	parent->allocated = 1;

	child = parent->children;
	while(child != (linked_list *) 0){
		child_struct = ((CUDA_child *) child->value)->struct_var;
		child_pointer = ((CUDA_child *) child->value)->pointer;

		((CUDA_child *) child->value)->old_pointer = *child_pointer;
		CUDA_struct_compile(child_struct, child_pointer);

		child = child->next;
	}

	cudaMemcpy(parent->value, parent->old_value, parent->size, cudaMemcpyHostToDevice);
	*pointer = parent->value;
}

void *CUDA_struct_free(CUDA_struct *parent){
	linked_list *child;
	linked_list *next_child;
	void **child_pointer;
	void *old_pointer;
	void *output;
	CUDA_struct *child_struct;
	
	if(!parent->allocated){
		fprintf(stderr, "Error: the CUDA_struct was never compiled. Compile it with void *new_struct = CUDA_struct_compile(my_CUDA_struct_pointer);");
		exit(1);
	}

	if(parent->freeing){
		return parent->old_value;
	}

	parent->freeing = 1;
	child = parent->children;
	while(child != (linked_list *) 0){
		old_pointer = ((CUDA_child *) child->value)->old_pointer;
		child_pointer = ((CUDA_child *) child->value)->pointer;
		child_struct = ((CUDA_child *) child->value)->struct_var;
		
		CUDA_struct_free(child_struct);

		*child_pointer = old_pointer;

		free((CUDA_child *) child->value);
		next_child = child->next;
		free(child);
		child = next_child;
	}

	cudaMemcpy(parent->old_value, parent->value, parent->size, cudaMemcpyDeviceToHost);

	output = parent->old_value;
	cudaFree(parent->value);
	return output;
}

__global__ void CUDA_test(struct test *t, int *t_ans, int *t2_ans){
	*t_ans = t->x + t->child->x;
	*t2_ans = t->child->parent->child->x;
}

int main(){
	struct test *t;
	struct test *t2;

	printf("Creating structs...\n");

	t = (struct test *) malloc(sizeof(struct test));
	t2 = (struct test *) malloc(sizeof(struct test));

	printf("Initializing gpu variables...\n");

	cudaMalloc(&t_ans, sizeof(int *));
	cudaMalloc(&t2_ans, sizeof(int *));
	cudaMemset(t_ans, 0, sizeof(int));
	cudaMemset(t2_ans, 0, sizeof(int));
	CUDA_struct *t_CUDA;
	CUDA_struct *t2_CUDA;
	
	printf("Creating CUDA_structs...\n");

	t_CUDA = create_CUDA_struct(t, sizeof(struct test));
	t2_CUDA = create_CUDA_struct(t2, sizeof(struct test));

	t->x = 2;
	t2->x = 3;

	printf("Adding t2 as child of t...\n");

	t->child = t2;
	CUDA_struct_add_child(t_CUDA, t2_CUDA, (void **) &(t->child));
	t2->child = (struct test *) 0;

	printf("Adding t as child of t2...\n");

	t2->parent = t;
	CUDA_struct_add_child(t2_CUDA, t_CUDA, (void **) &(t2->parent));
	
	printf("Compiling CUDA_struct...\n");

	CUDA_struct_compile(t_CUDA, (void **) &t);

	printf("Running test kernel...\n");

	CUDA_test<<<1, 1>>>(t, t_ans, t2_ans);

	int t_device;
	int t2_device;

	printf("Copying memory back to host...\n");

	cudaMemcpy((void *) &t_device, t_ans, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *) &t2_device, t2_ans, sizeof(int), cudaMemcpyDeviceToHost);
	
	printf("Freeing gpu variables...\n");

	cudaFree(t_ans);
	cudaFree(t2_ans);	

	printf("Freeing CUDA_struct...\n");

	t = (test *) CUDA_struct_free(t_CUDA);

	printf("Printing last error...\n");

	cudaDeviceSynchronize();

	cudaError_t err;
	err = cudaGetLastError();
	printf("CUDA error: %s\n", cudaGetErrorString(err));

	printf("Test results: %d %d\n", t_device, t2_device);
	printf("Expected    : 5 3\n");

	printf("Freeing t and t2...\n");

	free(t);
	free(t2);
	printf("done\n");
	return 0;
}
