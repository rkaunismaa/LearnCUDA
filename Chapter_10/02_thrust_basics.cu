/*
 * Chapter 10 — 02_thrust_basics.cu
 *
 * Demonstrates the Thrust library (header-only STL-like GPU algorithms):
 *   - device_vector and host_vector
 *   - fill, sequence, transform
 *   - reduce, min_element, max_element
 *   - sort
 *   - transform_reduce
 *   - copy_if (filter)
 *
 * Thrust is header-only — no -lthrust needed.
 *
 * Compile:
 *   nvcc -arch=sm_89 -O2 -o thrust_basics 02_thrust_basics.cu
 * Run:
 *   ./thrust_basics
 */

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <math.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = (call);                                          \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_err));          \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Custom functor: square a float (for use in transform)
struct Square {
    __host__ __device__
    float operator()(float x) const { return x * x; }
};

// Custom functor: is the value positive?
struct IsPositive {
    __host__ __device__
    bool operator()(float x) const { return x > 0.0f; }
};

int main()
{
    const int N = 1 << 20;  // 1M elements
    printf("Thrust Basics Demo: N = %d\n\n", N);

    // ================================================================
    // 1. device_vector and host_vector
    // ================================================================
    printf("--- 1. device_vector / host_vector ---\n");
    {
        // Create device vector filled with zeros
        thrust::device_vector<float> d_vec(N, 0.0f);

        // Create from host data
        thrust::host_vector<float> h_src(N);
        for (int i = 0; i < N; i++) h_src[i] = (float)i / N;
        thrust::device_vector<float> d_src = h_src;  // Implicit H2D copy

        // Copy back to host
        thrust::host_vector<float> h_dst = d_src;    // Implicit D2H copy
        printf("h_dst[0]=%.4f  h_dst[N/2]=%.4f  h_dst[N-1]=%.4f\n",
               h_dst[0], h_dst[N/2], h_dst[N-1]);
    }

    // ================================================================
    // 2. fill and sequence
    // ================================================================
    printf("\n--- 2. fill / sequence ---\n");
    {
        thrust::device_vector<float> d(10);
        thrust::fill(d.begin(), d.end(), 3.14f);
        printf("After fill(3.14): [%.2f, %.2f, ...]\n",
               (float)d[0], (float)d[1]);

        thrust::sequence(d.begin(), d.end(), 1.0f, 2.0f);  // start=1, step=2
        printf("After sequence(start=1,step=2): [%.0f, %.0f, %.0f, ...]\n",
               (float)d[0], (float)d[1], (float)d[2]);
    }

    // ================================================================
    // 3. transform
    // ================================================================
    printf("\n--- 3. transform (square each element) ---\n");
    {
        thrust::device_vector<float> d_in(8);
        thrust::sequence(d_in.begin(), d_in.end(), 1.0f);  // 1,2,3,...8

        thrust::device_vector<float> d_out(8);
        thrust::transform(d_in.begin(), d_in.end(), d_out.begin(), Square());

        printf("Input:  [1, 2, 3, 4, 5, 6, 7, 8]\n");
        printf("Output: [");
        for (int i = 0; i < 8; i++)
            printf("%.0f%s", (float)d_out[i], i < 7 ? ", " : "]\n");
    }

    // ================================================================
    // 4. reduce, min/max
    // ================================================================
    printf("\n--- 4. reduce / min_element / max_element ---\n");
    {
        thrust::device_vector<float> d(N);
        thrust::sequence(d.begin(), d.end(), 1.0f);  // 1,2,...,N

        float sum = thrust::reduce(d.begin(), d.end(), 0.0f, thrust::plus<float>());
        printf("Sum of 1..%d = %.0f (expected %.0f)\n",
               N, sum, (float)N * (N + 1) / 2);

        auto min_it = thrust::min_element(d.begin(), d.end());
        auto max_it = thrust::max_element(d.begin(), d.end());
        printf("Min: %.0f  Max: %.0f\n",
               (float)*min_it, (float)*max_it);
    }

    // ================================================================
    // 5. sort
    // ================================================================
    printf("\n--- 5. sort ---\n");
    {
        thrust::host_vector<float> h(10);
        h[0]=3; h[1]=1; h[2]=4; h[3]=1; h[4]=5;
        h[5]=9; h[6]=2; h[7]=6; h[8]=5; h[9]=3;
        thrust::device_vector<float> d = h;

        printf("Before: [");
        for (int i = 0; i < 10; i++) printf("%.0f%s", (float)d[i], i<9?", ":""]);
        printf("]\n");

        thrust::sort(d.begin(), d.end());

        printf("After:  [");
        for (int i = 0; i < 10; i++) printf("%.0f%s", (float)d[i], i<9?", ":""]);
        printf("]\n");
    }

    // ================================================================
    // 6. transform_reduce (sum of squares = L2 norm squared)
    // ================================================================
    printf("\n--- 6. transform_reduce (L2 norm) ---\n");
    {
        thrust::device_vector<float> d(N);
        thrust::sequence(d.begin(), d.end(), 1.0f);

        // Compute sum of squares in one pass: no intermediate storage needed
        float sum_sq = thrust::transform_reduce(
            d.begin(), d.end(),
            Square(),          // Transform: x → x²
            0.0f,              // Initial value
            thrust::plus<float>()  // Reduce: sum
        );
        printf("Sum of squares (1..%d): %.0f\n", N, sum_sq);
        printf("L2 norm: %.1f\n", sqrtf(sum_sq));
    }

    // ================================================================
    // 7. copy_if (filter positive elements)
    // ================================================================
    printf("\n--- 7. copy_if (filter) ---\n");
    {
        thrust::host_vector<float> h(10);
        for (int i = 0; i < 10; i++) h[i] = (float)(i - 5);  // -5,-4,...,4
        thrust::device_vector<float> d_in = h;
        thrust::device_vector<float> d_out(10);

        auto end = thrust::copy_if(d_in.begin(), d_in.end(),
                                    d_out.begin(), IsPositive());
        int count = end - d_out.begin();

        printf("Input:    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]\n");
        printf("Positive: [");
        for (int i = 0; i < count; i++)
            printf("%.0f%s", (float)d_out[i], i < count-1 ? ", " : "]\n");
        printf("Count: %d\n", count);
    }

    return 0;
}
