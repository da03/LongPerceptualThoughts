

# Synthesizing LongPerceptualThoughts

Our data synthesis pipeline consists of three stages:

1. **Stage 1:** Generate multiple-choice questions using a large language model (LLM).
2. **Stage 2:** Generate short chain-of-thoughts using a vision-language model (VLM).
3. **Stage 3:** Expand short chain-of-thoughts into long reasoning traces using a reasoning LLM.

To run all three stages sequentially, use the provided script:

```bash
./run_3_stages_test.sh
```

Stages 2 and 3 process inputs in chunks to support parallel job execution across multiple GPUs. You can adjust chunk sizes by modifying the `STAGE_2_CHUNK_SIZE` and `STAGE_3_CHUNK_SIZE` variables in the script.

