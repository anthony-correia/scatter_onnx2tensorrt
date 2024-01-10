#!/usr/bin/env python3
import os
import os.path as op
import argparse
import subprocess

import numpy as np
import torch
import torch_scatter
import onnx
import onnxruntime


class ScatterModule(torch.nn.Module):
    """A PyTorch module involving a scatter sum. To be converted to ONNX."""

    def __init__(self, n_outputs: int, n_features: int):
        super(ScatterModule, self).__init__()
        self.n_outputs = int(n_outputs)
        self.network = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_features, out_features=n_features),
            torch.nn.ReLU(),
        )

    def forward(self, e: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Scatter-sum ``e`` with indices ``index``.

        Args:
            e: 2D tensor of shape ``(n_e, dim_size)``
            index: 1D tensor of shape ``(n_e,)`` of integer indices

        Returns:
            Tensor of shape ``(n_outputs, dim_size)``
            The result of scattering elements in ``e`` along dimension 0, according
            to the indices in ``index``, with a sum reduction.
            ``index`` is broadcasted along the dimension 1 of ``e``.
        """
        # If no network,
        # TensorRT fails with `CUDA error: an illegal memory access was encountered`,
        # so I added some network applied to `e`.
        e = self.network(e)
        output = torch.zeros(
            size=(self.n_outputs, dim_size), device=e.device, dtype=e.dtype
        )
        torch_scatter.scatter(
            src=e,
            # broadcasting (should be done automatically anyway)
            index=index.unsqueeze(-1).expand(-1, e.shape[1]),
            dim=0,
            reduce="sum",
            out=output,
        )
        return output


def get_parsed_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Check ONNX -> TensorRT conversion for scatter_sum."
    )
    parser.add_argument(
        "--n_indices", type=int, default=1000, help="Number of indices and `e` inputs."
    )
    parser.add_argument(
        "--dim_size", type=int, default=3, help="Dimension of `e` input."
    )
    parser.add_argument("--n_outputs", type=int, default=100, help="Number of outputs.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "-o", "--output_path", help="Path where to save the ONNX file", default=None
    )

    return parser.parse_args()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get parsed arguments
    parsed_args = get_parsed_args()
    n_indices: int = parsed_args.n_indices
    dim_size: int = parsed_args.dim_size
    n_outputs: int = parsed_args.n_outputs
    seed: int = parsed_args.seed
    output_path: str | None = parsed_args.output_path

    # For reproducibility
    torch.random.manual_seed(seed)  # for reproducibility

    # Define random input tensors
    e_dummy = torch.randn(size=(n_indices, dim_size), device=device)
    index_dummy = torch.randint(high=n_outputs, size=(n_indices,), device=device)

    # Instantiate module
    print("Instantiate custom PyTorch `ScatterModule` model.")
    scatterModule = (
        ScatterModule(n_outputs=n_outputs, n_features=dim_size).to(device).eval()
    )

    # Compute output according to PyTorch
    dummy_inputs = (e_dummy, index_dummy)
    with torch.no_grad():
        output_pytorch = scatterModule(*dummy_inputs)

    # Define where to save the ONNX model
    if output_path is None:
        output_path = op.join(
            op.abspath(op.dirname(__file__)),
            "onnx",
            f"{n_indices}_{dim_size}_{n_outputs}_{seed}.onnx",
        )

    os.makedirs(op.dirname(output_path), exist_ok=True)  # Create output directory

    print("Convert model to ONNX.")
    torch.onnx.export(
        model=scatterModule,
        args=dummy_inputs,
        f=output_path,
        input_names=("e", "index"),
        output_names=("output",),
        opset_version=17,
    )
    print(f"Model was saved in {output_path}.")

    print("Check model integrity.")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model, full_check=True)

    print("Run ONNX Inference.")
    ort_session = onnxruntime.InferenceSession(output_path)
    (output_onnx,) = ort_session.run(
        [ort_session.get_outputs()[0].name],
        input_feed={
            input_node.name: input_tensor.cpu().numpy()
            for (input_node, input_tensor) in zip(
                ort_session.get_inputs(), dummy_inputs
            )
        },
    )

    if np.allclose(output_onnx, output_pytorch.cpu().numpy(), rtol=1e-03, atol=1e-05):
        print("ONNX and PyTorch inference yielded the same output.")
    else:
        raise RuntimeError(
            "ONNX inference yields a different output than PyTorch inference. "
            "That is unexpected!"
        )

    trt_exec_command = ["trtexec", f"--onnx={output_path}"]
    print("Running " + " ".join(trt_exec_command))
    result = subprocess.run(trt_exec_command)

    if result.returncode == 0:
        print("TensorRT conversion has been successful.")
    else:
        print("TensorRT conversion has failed.")
