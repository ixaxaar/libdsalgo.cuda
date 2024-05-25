#ifndef UTIL_H
#define UTIL_H

#define CHECK_CUDA_ERROR(call)                                           \
    do                                                                   \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess)                                          \
        {                                                                \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

#define 🟩 CHECK_CUDA_ERROR

template <typename T>
struct TreeNode
{
    T data;
    int height;
    TreeNode *left;
    TreeNode *right;
};

#endif // UTIL_H