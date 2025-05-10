# VideoToASCII Performance Test Results

This document summarizes the performance testing results for the VideoToASCII converter, comparing non-parallel (sequential) processing with various parallel processing configurations.

## Test Configuration

- **Test Videos**: small (320x240), medium (640x480), and large (1280x720)
- **Process Counts**: 1 (sequential baseline), 2, 11, and 22 processes
- **Batch Sizes**: 1, 5, 10, 20, and 50 frames per batch
- **System**: 22 CPU cores

## Summary of Results

| Video Size | Sequential Time | Best Parallel Time | Best Configuration | Speedup | Time Reduction |
|------------|-----------------|-------------------|-------------------|---------|----------------|
| Small      | 7.72s           | 4.02s             | 11 processes, batch size 20 | 1.92x | 3.70s (48.0%) |
| Medium     | 7.39s           | 3.43s             | 11 processes, batch size 50 | 2.16x | 3.96s (53.6%) |
| Large      | 8.89s           | 3.71s             | 11 processes, batch size 50 | 2.39x | 5.17s (58.2%) |

## Key Findings

1. **Optimal Process Count**: Using approximately half of the available CPU cores (11 out of 22) consistently provided the best performance. Using all available cores (22) actually resulted in slightly worse performance, likely due to overhead from context switching and resource contention.

2. **Optimal Batch Size**: Larger batch sizes (20-50) generally performed better than smaller batch sizes, especially when combined with a moderate number of processes. This suggests that the overhead of starting and managing processes is significant, and processing more frames per batch helps amortize this cost.

3. **Scaling with Video Size**: The speedup from parallelization increases with video size. The large video (1280x720) showed the greatest benefit from parallelization with a 2.39x speedup, while the small video (320x240) showed a more modest 1.92x speedup.

4. **Quality Preservation**: All parallel configurations maintained perfect similarity (1.0) with the sequential baseline, indicating that parallelization does not affect the quality of the output.

5. **Diminishing Returns**: Using more than half of the available CPU cores showed diminishing returns and sometimes even decreased performance. This suggests that the optimal configuration balances parallelism with resource contention.

## Detailed Charts

The following charts are available in the `test_output/plots` directory:

1. **Execution Time vs. Process Count**: Shows absolute execution times for different process counts and batch sizes, with the sequential baseline highlighted.
2. **Speedup vs. Process Count**: Shows the speedup factor relative to the sequential baseline.
3. **Speedup vs. Batch Size**: Shows how speedup varies with batch size for different process counts.
4. **Quality vs. Performance**: Visualizes the relationship between speedup and output quality.
5. **Sequential vs. Parallel Comparison**: Direct comparison between sequential processing and the best parallel configuration.

## Conclusion

Parallelization significantly improves the performance of the VideoToASCII converter, with the best configuration achieving a speedup of up to 2.39x over sequential processing. The optimal configuration uses approximately half of the available CPU cores with a relatively large batch size (20-50 frames). This configuration balances the benefits of parallelism with the overhead of process management and resource contention.

For best results on a system with multiple CPU cores, it is recommended to use:
- Process count: Approximately half of the available CPU cores
- Batch size: 20-50 frames per batch

These settings provide a good balance between performance and resource utilization across different video sizes.