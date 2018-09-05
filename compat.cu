#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "compat.h"

int *t_ans;
int *t2_ans;

struct test{
	int x;
	struct test *child;
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

CUDA_struct *create_CUDA_struct(void **value, size_t size){
	CUDA_struct *output;
	output = (CUDA_struct *) malloc(sizeof(CUDA_struct));
	output->children = (linked_list *) 0;
	output->value = value;
	output->size = size;
	return output;
}

void CUDA_struct_add_child(CUDA_struct *parent, CUDA_struct *child){
	parent->children = add_linked_list(parent->children, create_linked_list((void *) child));
}

void *CUDA_struct_compile(CUDA_struct *parent){
	linked_list *child;

	child = parent->children;
	while(child != (linked_list *) 0){
		CUDA_struct_compile((CUDA_struct *) child->value);
		child = child->next;
	}

	parent->old_value = *(parent->value);
	cudaMalloc(parent->value, parent->size);
	cudaMemcpy(*(parent->value), parent->old_value, parent->size, cudaMemcpyHostToDevice);
	return *(parent->value);
}

void *CUDA_struct_free(CUDA_struct *parent){
	void *output;
	linked_list *child;
	linked_list *next_child;

	child = parent->children;
	while(child != (linked_list *) 0){
		CUDA_struct_free((CUDA_struct *) child->value);
		next_child = child->next;
		free(child);
		child = next_child;
	}

	cudaFree(*(parent->value));
	output = parent->old_value;
	*(parent->value) = parent->old_value;
	free(parent);
	return output;
}

__global__ void CUDA_test(struct test *t, int *t_ans, int *t2_ans){
	*t_ans = t->x + t->child->x;
	*t2_ans = t->child->x;
}

int main(){
	struct test *t;
	struct test *t2;

	t = (struct test *) malloc(sizeof(struct test));
	t2 = (struct test *) malloc(sizeof(struct test));

	cudaMalloc(&t_ans, sizeof(int *));
	cudaMalloc(&t2_ans, sizeof(int *));

	CUDA_struct *t_CUDA;
	CUDA_struct *t2_CUDA;
	
	t_CUDA = create_CUDA_struct((void **) &t, sizeof(struct test));
	t2_CUDA = create_CUDA_struct((void **) &(t->child), sizeof(struct test));

	t->x = 2;
	t2->x = 3;

	t->child = t2;
	CUDA_struct_add_child(t_CUDA, t2_CUDA);
	t2->child = (struct test *) 0;
	
	CUDA_struct_compile(t_CUDA);
	
	CUDA_test<<<1, 1>>>(t, t_ans, t2_ans);

	int t_device;
	int t2_device;

	cudaMemcpy((void *) &t_device, t_ans, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *) &t2_device, t2_ans, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(t_ans);
	cudaFree(t2_ans);	

	CUDA_struct_free(t_CUDA);

	free(t2);
	free(t);
	printf("%d %d\n", t_device, t2_device);
	return 0;
}
