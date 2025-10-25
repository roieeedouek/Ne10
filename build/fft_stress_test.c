#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "NE10.h"

#define FFT_SIZE 128
#define TEST_DURATION_SECONDS 5

int main(void)
{
    ne10_int32_t fft_size = FFT_SIZE;
    ne10_fft_r2c_cfg_float32_t cfg;
    ne10_float32_t *time_domain;
    ne10_fft_cpx_float32_t *freq_domain;

    // Initialize NE10
    if (ne10_init() != NE10_OK) {
        fprintf(stderr, "Failed to initialize NE10\n");
        return 1;
    }

    printf("NE10 FFT Stress Test\n");
    printf("====================\n");
    printf("FFT Size: %d points (Real FFT)\n", fft_size);
    printf("Test Duration: %d seconds\n\n", TEST_DURATION_SECONDS);

    // Allocate aligned memory for input and output
    time_domain = (ne10_float32_t*) NE10_MALLOC(fft_size * sizeof(ne10_float32_t));
    freq_domain = (ne10_fft_cpx_float32_t*) NE10_MALLOC((fft_size/2 + 1) * sizeof(ne10_fft_cpx_float32_t));

    if (!time_domain || !freq_domain) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize FFT configuration
    cfg = ne10_fft_alloc_r2c_float32(fft_size);
    if (!cfg) {
        fprintf(stderr, "FFT configuration failed\n");
        NE10_FREE(time_domain);
        NE10_FREE(freq_domain);
        return 1;
    }

    // Fill input with test data (sine wave)
    for (ne10_int32_t i = 0; i < fft_size; i++) {
        time_domain[i] = sin(2.0 * 3.14159265359 * 5.0 * i / fft_size);
    }

    printf("Starting stress test...\n");
    printf("Running FFTs continuously for %d seconds\n\n", TEST_DURATION_SECONDS);

    // Warm-up run
    for (int i = 0; i < 1000; i++) {
        ne10_fft_r2c_1d_float32_neon(freq_domain, time_domain, cfg);
    }

    // Get start time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Run FFTs for specified duration
    unsigned long long fft_count = 0;
    double elapsed_seconds;

    do {
        // Perform FFT
        ne10_fft_r2c_1d_float32_neon(freq_domain, time_domain, cfg);
        fft_count++;

        // Check elapsed time every 10000 iterations to reduce overhead
        if (fft_count % 10000 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &end);
            elapsed_seconds = (end.tv_sec - start.tv_sec) +
                             (end.tv_nsec - start.tv_nsec) / 1e9;
        }
    } while (elapsed_seconds < TEST_DURATION_SECONDS);

    // Final time measurement
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_seconds = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    // Calculate performance metrics
    double ffts_per_second = fft_count / elapsed_seconds;
    double samples_per_second = ffts_per_second * fft_size;
    double megasamples_per_second = samples_per_second / 1e6;
    double time_per_fft_us = (elapsed_seconds * 1e6) / fft_count;

    // Print results
    printf("Test Results:\n");
    printf("=============\n");
    printf("Total FFTs performed:    %llu\n", fft_count);
    printf("Total time:              %.3f seconds\n", elapsed_seconds);
    printf("FFTs per second:         %.2f FFT/s\n", ffts_per_second);
    printf("Time per FFT:            %.3f microseconds\n", time_per_fft_us);
    printf("Sample throughput:       %.2f MSamples/s\n", megasamples_per_second);
    printf("\nPeak Performance Summary:\n");
    printf("  - %.0f FFTs per second\n", ffts_per_second);
    printf("  - %.2f million samples per second\n", megasamples_per_second);

    // Verify output (sanity check)
    printf("\nSanity check - First few frequency bins:\n");
    for (int i = 0; i < 5; i++) {
        printf("  Bin %d: %.3f + %.3fi (magnitude: %.3f)\n",
               i, freq_domain[i].r, freq_domain[i].i,
               sqrt(freq_domain[i].r * freq_domain[i].r +
                    freq_domain[i].i * freq_domain[i].i));
    }

    // Cleanup
    NE10_FREE(cfg);
    NE10_FREE(time_domain);
    NE10_FREE(freq_domain);

    return 0;
}
