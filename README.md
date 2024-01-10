# `scatter_add` Conversion Issue in TensorRT

## Problem Description

Conversion of an ONNX model to TensorRT using `trtexec`,
which includes a `scatterElements` operation with a reduction like `"sum"`,
fails when the number of indices in the operation exceeds the output count.

Successful conversion requires `n_indices <= n_outputs`.

Consider the following PyTorch model snippet:
```python
import torch_scatter

n_indices: int = ...
dim_size: int = ...
n_outputs: int = ...

e_dummy = torch.randn(size=(n_indices, dim_size), device=device)
index_dummy = torch.randint(high=n_outputs, size=(n_indices,), device=device)

class ScatterModule(torch.nn.Module):
    def forward(self, e: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return torch_scatter.scatter(
            src=e,
            # broadcasting (should be done automatically anyway)
            index=index.unsqueeze(-1).expand(-1, e.shape[1]),
            dim=0,
            reduce="sum",
        )
```
Converting this model using `trtexec` triggers an assertion error:
```
Assertion failed: indicesDims.d[i] <= dataDims.d[i] && "Indices dimensions must be less than data dimensions!"
```
This error likely originates from [this line of the ONNX-TensorRT code](https://github.com/onnx/onnx-tensorrt/blob/bacfaaa951653cd4e72efe727a543567cb38f7de/onnx2trt_utils.cpp#L2100).

In the scenarios I've encountered within Graph Neural Networks, the number of indices
(`n_indices`, corresponding to the edges in the graph) is significantly
larger than the number of outputs (`n_outputs`, corresponding to the nodes in the graph).


## Prerequisites

- TensorRT, (I used `8.6.1.6-1+cuda11.8`).
- `trtexec` accessible in the PATH environment variable. For instance:
    ```bash
    PATH="/usr/src/tensorrt/bin:$PATH"
    ```

## Reproduce The Issue

The ONNX models are stored with the naming convention
`onnx/{n_indices}_{dim_size}_{n_outputs}_{seed}.onnx`.

To replicate the issue, execute the following commands:
```bash
# This command fails when `n_outputs = 100` and `n_indices = 1000`.
trtexec --onnx="1000_3_100_0.onnx"

# This command succeeds when `n_outputs` equals `n_indices` (both are 100).
trtexec --onnx="100_3_100_0.onnx"
```
 

## Reproducing the ONNX files

### Setting Up the Enviromnent

```bash
# Install miniforge of equivalent
# See https://github.com/conda-forge/miniforge#unix-like-platforms-mac-os--linux
# For instance:
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# Create and activate the environment from `environment.yaml`
conda env create -f environment.yaml

# Activate the environment
conda activate onnx2tensorrt_scatter_issue
```

### Running `run.py`

The `run.py` script:
1. Defines a PyTorch model with a `scatter_add` operation.
2. Convert it to ONNX without dynamic shapes. *(Utilizing dynamic shapes triggers
a different error, specifically an unsupported layer exception,
rather than the previously mentioned assertion error.)*
3. Verifies that ONNXRuntime and PyTorch inference produce identical outputs.
4. Run `trtexec --onnx=<corresponding_onnx_file.onnx>`

Running the script as follows demonstrates the issue:
```bash
# Fails for `n_outputs = 100` and `n_indices = 1000`.
./run.py --n_outputs 100 --n_indices 1000

# Succeeds for `n_outputs = n_indices = 100`.
./run.py --n_outputs 100 --n_indices 100
```

