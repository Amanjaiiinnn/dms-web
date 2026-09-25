"""Wrap a full-integer TFLite model with float32 input/output.

LiteRT.js (the browser runtime) does not accept INT8 input/output tensors.
This adds a QUANTIZE op (float32 -> int8) before the original input and a
DEQUANTIZE op (int8 -> float32) after the original output. The int8 network
itself is untouched, so the results are identical to the original model.

How models/phone_smoke.tflite was made from the board's INT8 export:

    pip install tensorflow
    python tools/make_float_io_model.py best_full_integer_quant.tflite models/phone_smoke.tflite
"""
import sys

from tensorflow.lite.python import schema_py_generated as schema
from tensorflow.lite.tools import flatbuffer_utils

QUANTIZE = schema.BuiltinOperator.QUANTIZE      # 114
DEQUANTIZE = schema.BuiltinOperator.DEQUANTIZE  # 6


def opcode_index(model, builtin, version):
    for i, oc in enumerate(model.operatorCodes):
        if max(oc.builtinCode, oc.deprecatedBuiltinCode) == builtin:
            oc.version = max(oc.version, version)
            return i
    oc = schema.OperatorCodeT()
    oc.builtinCode = builtin
    oc.deprecatedBuiltinCode = builtin if builtin < 127 else 127
    oc.version = version
    model.operatorCodes.append(oc)
    return len(model.operatorCodes) - 1


def new_float_tensor(model, sg, like, name):
    buf = schema.BufferT()
    model.buffers.append(buf)
    t = schema.TensorT()
    t.shape = list(like.shape)
    t.shapeSignature = list(like.shapeSignature) if like.shapeSignature is not None else None
    t.type = schema.TensorType.FLOAT32
    t.buffer = len(model.buffers) - 1
    t.name = name.encode()
    sg.tensors.append(t)
    return len(sg.tensors) - 1


def main(src, dst):
    model = flatbuffer_utils.read_model(src)
    sg = model.subgraphs[0]
    assert len(sg.inputs) == 1 and len(sg.outputs) == 1
    q_in, q_out = sg.inputs[0], sg.outputs[0]
    assert sg.tensors[q_in].type == schema.TensorType.INT8
    assert sg.tensors[q_out].type == schema.TensorType.INT8

    f_in = new_float_tensor(model, sg, sg.tensors[q_in], "images")
    f_out = new_float_tensor(model, sg, sg.tensors[q_out], "output")

    quant = schema.OperatorT()
    quant.opcodeIndex = opcode_index(model, QUANTIZE, 2)
    quant.inputs, quant.outputs = [f_in], [q_in]
    quant.builtinOptionsType = schema.BuiltinOptions.QuantizeOptions
    quant.builtinOptions = schema.QuantizeOptionsT()

    dequant = schema.OperatorT()
    dequant.opcodeIndex = opcode_index(model, DEQUANTIZE, 2)
    dequant.inputs, dequant.outputs = [q_out], [f_out]
    dequant.builtinOptionsType = schema.BuiltinOptions.DequantizeOptions
    dequant.builtinOptions = schema.DequantizeOptionsT()

    sg.operators.insert(0, quant)
    sg.operators.append(dequant)
    sg.inputs, sg.outputs = [f_in], [f_out]

    for sig in model.signatureDefs or []:
        for t in sig.inputs:
            if t.tensorIndex == q_in:
                t.tensorIndex = f_in
        for t in sig.outputs:
            if t.tensorIndex == q_out:
                t.tensorIndex = f_out

    flatbuffer_utils.write_model(model, dst)
    print("wrote", dst)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
