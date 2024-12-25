#! /usr/bin/env python

import onnx
import onnxruntime as ort
import numpy as np
import os

## Model creation
matrix_in1 = onnx.helper.make_tensor_value_info("matrix_in1", onnx.TensorProto.FLOAT, [2, 3])
matrix_in2 = onnx.helper.make_tensor_value_info("matrix_in2", onnx.TensorProto.FLOAT, [3, 2])
matrix_out = onnx.helper.make_tensor_value_info("matrix_out", onnx.TensorProto.FLOAT, [2, 2])

helper_node = onnx.helper.make_node("MatMul", ["matrix_in1", "matrix_in2"], ["matrix_out"])
helper_graph = onnx.helper.make_graph([helper_node], "MatMulGraph", [matrix_in1, matrix_in2], [matrix_out])

test_model = onnx.helper.make_model(helper_graph)
onnx.save(test_model, "matmul.onnx")

## Model inference
onnx_session = ort.InferenceSession("matmul.onnx", providers=["MIGraphXExecutionProvider", "ROCMExecutionProvider"])
onnx.checker.check_model(test_model)

matrix_in1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
matrix_in2 = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
output_matrix = onnx_session.run(None, {"matrix_in1": matrix_in1, "matrix_in2": matrix_in2})

print(f"MATRIX A\n {matrix_in1}\n\nMATRIX B\n {matrix_in2}\n\nMATRIX AB\n {output_matrix[0]}")

os.remove("matmul.onnx")
