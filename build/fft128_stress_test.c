#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "NE10.h"

#define FFT_SIZE 128
#define TEST_DURATION_SECONDS 10

int main(void)
{
    ne10_int32_t fft_size = FFT_SIZE;
    ne10_fft_cfg_float32_t cfg;
    ne10_fft_cpx_float32_t *input;
    ne10_fft_cpx_float32_t *output;

    // Initialize NE10
    if (ne10_init() != NE10_OK) {
        fprintf(stderr, "Failed to initialize NE10\n");
        return 1;
    }

    printf("===========================================\n");
    printf("NE10 128-Point Complex FFT Stress Test\n");
    printf("===========================================\n");
    printf("FFT Size: %d points (Complex FFT)\n", fft_size);
    printf("Test Duration: %d seconds\n", TEST_DURATION_SECONDS);
    printf("Target: Maximum FFT throughput\n\n");

    // Allocate aligned memory for input and output
    input = (ne10_fft_cpx_float32_t*) NE10_MALLOC(fft_size * sizeof(ne10_fft_cpx_float32_t));
    output = (ne10_fft_cpx_float32_t*) NE10_MALLOC(fft_size * sizeof(ne10_fft_cpx_float32_t));

    if (!input || !output) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize FFT configuration
    cfg = ne10_fft_alloc_c2c_float32(fft_size);
    if (!cfg) {
        fprintf(stderr, "FFT configuration failed\n");
        NE10_FREE(input);
        NE10_FREE(output);
        return 1;
    }

    // Fill input with test data (complex sine wave)
    for (ne10_int32_t i = 0; i < fft_size; i++) {
        input[i].r = cos(2.0 * M_PI * 5.0 * i / fft_size);
        input[i].i = sin(2.0 * M_PI * 5.0 * i / fft_size);
    }

    printf("Warming up...\n");
    // Warm-up run
    for (int i = 0; i < 10000; i++) {
        ne10_fft_c2c_1d_float32_neon(output, input, cfg, 0);
    }

    printf("Starting stress test...\n");
    printf("Running continuous FFTs for %d seconds...\n\n", TEST_DURATION_SECONDS);

    // Get start time
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Run FFTs for specified duration
    unsigned long long fft_count = 0;
    double elapsed_seconds = 0;

    do {
        // Perform FFT
        ne10_fft_c2c_1d_float32_neon(output, input, cfg, 0);
        fft_count++;

        // Check elapsed time every 100000 iterations to reduce overhead
        if (fft_count % 100000 == 0) {
            clock_gettime(CLOCK_MONOTONIC, &end);
            elapsed_seconds = (end.tv_sec - start.tv_sec) +
                             (end.tv_nsec - start.tv_nsec) / 1e9;

            // Print progress
            double current_rate = fft_count / elapsed_seconds;
            printf("\rProgress: %.1fs | FFTs: %llu | Rate: %.0f FFT/s",
                   elapsed_seconds, fft_count, current_rate);
            fflush(stdout);
        }
    } while (elapsed_seconds < TEST_DURATION_SECONDS);

    // Final time measurement
    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_seconds = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("\n\n");

    // Calculate performance metrics
    double ffts_per_second = fft_count / elapsed_seconds;
    double samples_per_second = ffts_per_second * fft_size;
    double megasamples_per_second = samples_per_second / 1e6;
    double time_per_fft_us = (elapsed_seconds * 1e6) / fft_count;
    double time_per_fft_ns = (elapsed_seconds * 1e9) / fft_count;

    // Print results
    printf("===========================================\n");
    printf("PERFORMANCE RESULTS\n");
    printf("===========================================\n");
    printf("Total FFTs performed:     %llu\n", fft_count);
    printf("Total time:               %.3f seconds\n", elapsed_seconds);
    printf("\n");
    printf("FFT Throughput:           %.0f FFT/s\n", ffts_per_second);
    printf("Time per FFT:             %.3f μs (%.0f ns)\n", time_per_fft_us, time_per_fft_ns);
    printf("Sample throughput:        %.2f MSamples/s\n", megasamples_per_second);
    printf("\n");
    printf("===========================================\n");
    printf("SUMMARY\n");
    printf("===========================================\n");
    printf("Peak Performance: %.0f FFTs per second\n", ffts_per_second);
    printf("Average Latency:  %.3f microseconds per FFT\n", time_per_fft_us);
    printf("Data Throughput:  %.2f million complex samples/s\n", megasamples_per_second);
    printf("===========================================\n");

    // Verify output (sanity check)
    printf("\nSanity Check - First 5 frequency bins:\n");
    for (int i = 0; i < 5; i++) {
        double magnitude = sqrt(output[i].r * output[i].r + output[i].i * output[i].i);
        printf("  Bin %d: %.3f + %.3fi (mag: %.3f)\n",
               i, output[i].r, output[i].i, magnitude);
    }

    // Cleanup
    NE10_FREE(cfg);
    NE10_FREE(input);
    NE10_FREE(output);

    printf("\nTest completed successfully!\n");
    return 0;
}
