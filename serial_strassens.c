/*
  Original code written by Henrik Almer - https://github.com/henrikalmer
  ---Added dynamic allocation of memory to matrices---
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef float matrix_type;
#define FORMAT "%f\t"

/*
allocate_matrix() is a function to allocate the matrix onto heap storage
the matrix is an array of pointers that each have an array of pointers
*/
matrix_type ** allocate_matrix(int size) 
{
    matrix_type **matrix_rows = (matrix_type**) malloc(sizeof (matrix_type *) * size);
    assert (matrix_rows != NULL); 
    for (int i = 0; i < size; ++i)
    {
        matrix_rows[i] = (matrix_type *) malloc(sizeof(matrix_type) * size);
        assert (matrix_rows[i] != NULL);
    }
	return matrix_rows;
}

/*
deallocate_matrix() is a function which deallocates a matrix once it's use is over
*/

void deallocate_matrix(matrix_type ** m, int size)
{
    // printf("in deall\n");
    for (int i=0; i<size; ++i)
    {
        free(m[i]);
    }
    free (m);
}

/* Takes a pointer to a 2-dimensional array matrix_type matrix of int size and
 * prints it to stdout, preceeded by a char label.
 */
void print_matrix(char *label, int size, matrix_type **matrix)
{
    printf("\n\n%s\n", label);
    printf("{\n");
    for(int i = 0; i < size; i++) {
        printf("\t[");
        for(int j = 0; j < size; j++) {
            printf(FORMAT, matrix[i][j]);
        }
        printf("]\n");
    }
    printf("}\n");
}

/* Takes a pointer to a 2-dimensional array matrix_type matrix of int size and
 * fills it with random numbers.
 */
void fill_matrix(int size, matrix_type **matrix)
{
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
}

/* Iterative matrix multiplication. The naive implementation.
 */
void naive_matrix_multiplication(int size,matrix_type **a , matrix_type **b , matrix_type **result)
{
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            result[i][j] = 0;
            for(int k = 0; k < size; k++) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

/* Subtract two matrices
 */
void subtract_matrices(int size, matrix_type **a , matrix_type **b , matrix_type **result)
{
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
}

/* Add two matrices
 */
void add_matrices(int size,matrix_type **a ,matrix_type **b , matrix_type **result)
{
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
}

/* Implementation of Strassen's recursive matrix multiplication
 * algorithm.
 */
void strassens_multiplication(int size,matrix_type **a , matrix_type **b , matrix_type **result)
{
    if(size <= 2) 
    {
        naive_matrix_multiplication(size, a, b, result);
    } 
    else 
    {
        int block_size = size / 2;

        matrix_type ** m1=allocate_matrix(block_size);
        matrix_type ** m2=allocate_matrix(block_size);
        matrix_type ** m3=allocate_matrix(block_size);
        matrix_type ** m4=allocate_matrix(block_size);
        matrix_type ** m5=allocate_matrix(block_size);
        matrix_type ** m6=allocate_matrix(block_size);
        matrix_type ** m7=allocate_matrix(block_size);
        matrix_type ** a11=allocate_matrix(block_size);
        matrix_type ** a12=allocate_matrix(block_size);
        matrix_type ** a21=allocate_matrix(block_size);
        matrix_type ** a22=allocate_matrix(block_size);
        matrix_type ** b11=allocate_matrix(block_size);
        matrix_type ** b12=allocate_matrix(block_size);
        matrix_type ** b21=allocate_matrix(block_size);
        matrix_type ** b22=allocate_matrix(block_size);
        matrix_type ** add_result1 = allocate_matrix(size);
        matrix_type ** add_result2 = allocate_matrix(size);
        matrix_type ** subtract_result = allocate_matrix(size);
        
        for(int i = 0; i < block_size; i++) {
            for(int j = 0; j < block_size; j++) {
                a11[i][j] = a[i][j];
                a12[i][j] = a[i][j + block_size];
                a21[i][j] = a[i + block_size][j];
                a22[i][j] = a[i + block_size][j + block_size];
                b11[i][j] = b[i][j];
                b12[i][j] = b[i][j + block_size];
                b21[i][j] = b[i + block_size][j];
                b22[i][j] = b[i + block_size][j + block_size];
            }
        }

        // m1
        add_matrices(block_size, a11, a22, add_result1);
        add_matrices(block_size, b11, b22, add_result2);
        strassens_multiplication(block_size, add_result1, add_result2,m1);
        // m2   
        add_matrices(block_size, a21, a22, add_result1);
        strassens_multiplication(block_size, add_result1, b11, m2);
        // m3
        subtract_matrices(block_size, b12, b22, subtract_result);
        strassens_multiplication(block_size, a11, subtract_result, m3);
        // m4
        subtract_matrices(block_size, b21, b11, subtract_result);
        strassens_multiplication(block_size, a22, subtract_result, m4);
        // m5
        add_matrices(block_size, a11, a12, add_result1);
        strassens_multiplication(block_size, add_result1, b22, m5);
        // m6
        subtract_matrices(block_size, a21, a11, subtract_result);
        add_matrices(block_size, b11, b12, add_result1);
        strassens_multiplication(block_size, subtract_result,
            add_result1, m6);
        // m7
        subtract_matrices(block_size, a12, a22, subtract_result);
        add_matrices(block_size, b21, b22, add_result1);
        strassens_multiplication(block_size, subtract_result,
            add_result1, m7);

        deallocate_matrix(a11, block_size);
        deallocate_matrix(a12, block_size);
        deallocate_matrix(a21, block_size);
        deallocate_matrix(a22, block_size);
        deallocate_matrix(b11, block_size);
        deallocate_matrix(b12, block_size);
        deallocate_matrix(b21, block_size);
        deallocate_matrix(b22, block_size);
        

        matrix_type ** c11=allocate_matrix(block_size);
        matrix_type ** c12=allocate_matrix(block_size);
        matrix_type ** c21=allocate_matrix(block_size);
        matrix_type ** c22=allocate_matrix(block_size);

        // c11
        add_matrices(block_size, m1, m4, add_result1);
        add_matrices(block_size, add_result1, m7, add_result2);
        subtract_matrices(block_size, add_result2, m5, c11);
        // c12
        add_matrices(block_size, m3, m5, c12);
        // c21
        add_matrices(block_size, m2, m4, c21);
        // c22
        add_matrices(block_size, m1, m3, add_result1);
        add_matrices(block_size, add_result1, m6, add_result2);
        subtract_matrices(block_size, add_result2, m2, c22);

        for(int k = 0; k < block_size; k++) {
            for(int j = 0; j < block_size; j++) {
                result[k][j] = c11[k][j];
                result[k][j + block_size] = c12[k][j];
                result[k + block_size][j] = c21[k][j];
                result[k + block_size][j + block_size] = c22[k][j];
            }
        }
        deallocate_matrix(c11, block_size);
        deallocate_matrix(c12, block_size);
        deallocate_matrix(c21, block_size);
        deallocate_matrix(c22, block_size);
        deallocate_matrix(m1, block_size);
        deallocate_matrix(m2, block_size);
        deallocate_matrix(m3, block_size);
        deallocate_matrix(m4, block_size);
        deallocate_matrix(m5, block_size);
        deallocate_matrix(m6, block_size);
        deallocate_matrix(m7, block_size);
        deallocate_matrix(add_result1, size);
        deallocate_matrix(add_result2, size);
    }
}

int main(int argc, char *argv[])
{
    int size = argc == 1 ? 128 : atoi(argv[1]);
    
    matrix_type **matrix_a = allocate_matrix(size);
    matrix_type **matrix_b = allocate_matrix(size);
    matrix_type **matrix_result = allocate_matrix(size);

    fill_matrix(size, matrix_a);
    fill_matrix(size, matrix_b);

    strassens_multiplication(size, matrix_a, matrix_b, matrix_result);
    return 0;
}
