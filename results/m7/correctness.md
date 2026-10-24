# M7 correctness report

Reference: `kernels.prefix_attn.reference_packed` in fp64. Inputs ~N(0,1). Tolerance = max |err| over
all outputs: fp32 ≤ 2e-5, fp16 ≤ 2e-3. 'torch same-dtype' is the error of the PyTorch reference evaluated
in the input dtype (fp16 inputs are computed in fp32 by the reference), given for context.

## triton (TRITON_INTERPRET=1, Linux container)

- result file: `results/m7/correctness_triton.json` (commit `2ea0e9c170`)
- **92/92 cases pass**

| dtype | cases | worst max-abs-err | median max-abs-err | worst torch same-dtype |
|---|---|---|---|---|
| fp32 | 46 | 9.38e-07 | 4.76e-07 | 9.38e-07 |
| fp16 | 46 | 9.61e-04 | 5.41e-04 | 9.61e-04 |

## metal (Apple GPU (MPS))

- result file: `results/m7/correctness_metal.json` (commit `2ea0e9c170`)
- **92/92 cases pass**

| dtype | cases | worst max-abs-err | median max-abs-err | worst torch same-dtype |
|---|---|---|---|---|
| fp32 | 46 | 1.23e-06 | 5.68e-07 | 9.43e-07 |
| fp16 | 46 | 9.61e-04 | 4.81e-04 | 9.61e-04 |
