import pytest
import tvm
from tvm import relax, tirx
from tvm.ir import assert_structural_equal

n = tirx.Var("n", "int64")
c = tirx.Var("c", "int64")
h = tirx.Var("h", "int64")
w = tirx.Var("w", "int64")

def _make_sinfo(shape, dtype, ndim):
    vdevice = None
    return relax.TensorStructInfo(shape, dtype, vdevice, ndim if shape is None else -1)

def _check_tensor_sinfo(sinfo, expected_dtype, expected_shape, expected_ndim):
    """Asserts dtype, and either structural shape equality or correct ndim."""
    assert isinstance(sinfo, relax.TensorStructInfo)
    assert sinfo.dtype == expected_dtype
    if expected_shape is None:
        assert sinfo.shape is None
        assert sinfo.ndim == expected_ndim
    else:
        assert sinfo.shape is not None
        assert_structural_equal(sinfo.shape, relax.ShapeExpr(expected_shape))

@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "a_shape, b_shape, ndim, expected_shape, expected_ndim",
    [
        # Concrete Shapes
        ((1, 3, 224, 224), (1, 3, 224, 224), 4, (1, 3, 224, 224), 4),
        # Concrete Broadcast
        ((1, 3, 224, 224), (1, 3, 1, 1), 4, (1, 3, 224, 224), 4),
        # Symbolic Shapes
        ((n, c, h, w), (n, c, h, w), 4, (n, c, h, w), 4),
        # Symbolic Broadcast
        ((n, c, h, w), (1, c, 1, 1), 4, (n, c, h, w), 4),
        # Omitted Shape (Known Rank)
        (None, None, 4, None, 4),
        # Omitted Shape (Unknown Rank)
        (None, None, -1, None, -1),
    ]
)
def test_qnn_add_inference(dtype, a_shape, b_shape, ndim, expected_shape, expected_ndim):
    bb = relax.BlockBuilder()

    a = relax.Var("a", _make_sinfo(a_shape, dtype, ndim))
    a_scale = relax.Var("a_scale", relax.TensorStructInfo((), "float32"))
    a_zp = relax.Var("a_zp", relax.TensorStructInfo((), dtype))

    b = relax.Var("b", _make_sinfo(b_shape, dtype, ndim))
    b_scale = relax.Var("b_scale", relax.TensorStructInfo((), "float32"))
    b_zp = relax.Var("b_zp", relax.TensorStructInfo((), dtype))

    c_scale = relax.Var("c_scale", relax.TensorStructInfo((), "float32"))
    c_zp = relax.Var("c_zp", relax.TensorStructInfo((), dtype))

    with bb.function("main", params=[a, a_scale, a_zp, b, b_scale, b_zp, c_scale, c_zp]):
        out = relax.op.qnn.add(a, a_scale, a_zp, b, b_scale, b_zp, c_scale, c_zp)
        bb.emit_func_output(out)

    func = bb.get()["main"]
    _check_tensor_sinfo(func.ret_struct_info, dtype, expected_shape, expected_ndim)

@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "x_shape, w_shape, ndim, expected_shape, expected_ndim",
    [
        # Concrete
        ((1, 64, 56, 56), (128, 64, 3, 3), 4, (1, 128, 28, 28), 4),
        # Symbolic: out_dim = (in_dim + pad - kernel) // stride + 1
        # For H/W: (dim + 2 - 3) // 2 + 1  ->  (dim - 1) // 2 + 1
        ((n, 64, h, w), (128, 64, 3, 3), 4, (n, 128, (h - 1) // 2 + 1, (w - 1) // 2 + 1), 4),
        # Omitted (Known Rank)
        (None, None, 4, None, 4),
        # Omitted (Unknown Rank)
        (None, None, -1, None, -1),
    ]
)
def test_qnn_conv2d_inference(use_bias, dtype, x_shape, w_shape, ndim, expected_shape, expected_ndim):
    bb = relax.BlockBuilder()

    x = relax.Var("x", _make_sinfo(x_shape, dtype, ndim))
    x_scale = relax.Var("x_scale", relax.TensorStructInfo((), "float32"))
    x_zp = relax.Var("x_zp", relax.TensorStructInfo((), dtype))

    w = relax.Var("w", _make_sinfo(w_shape, dtype, ndim))
    w_scale = relax.Var("w_scale", relax.TensorStructInfo((), "float32"))
    w_zp = relax.Var("w_zp", relax.TensorStructInfo((), dtype))

    y_scale = relax.Var("y_scale", relax.TensorStructInfo((), "float32"))
    y_zp = relax.Var("y_zp", relax.TensorStructInfo((), dtype))

    # Initialize optional bias
    if use_bias:
        if expected_shape is not None:
            # Broadcastable bias shape: (out_channels, 1, 1)
            b_shape = (expected_shape[1], 1, 1)
            b_ndim = 3
        else:
            b_shape = None
            b_ndim = expected_ndim
        # Bias in QNN operations is typically int32 to hold the accumulated sum
        B = relax.Var("B", _make_sinfo(b_shape, "int32", b_ndim))
    else:
        B = None

    params = [x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp]
    if B is not None:
        params.append(B)

    with bb.function("main", params=params):
        out = relax.op.qnn.conv2d(
            x, x_scale, x_zp,
            w, w_scale, w_zp,
            y_scale, y_zp,
            B=B,
            strides=(2, 2), padding=(1, 1, 1, 1)
        )
        bb.emit_func_output(out)

    func = bb.get()["main"]
    _check_tensor_sinfo(func.ret_struct_info, dtype, expected_shape, expected_ndim)


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "x_shape, w_shape, ndim, expected_shape, expected_ndim",
    [
        # Concrete
        ((32, 128), (128, 256), 2, (32, 256), 2),
        # Symbolic
        ((n, 128), (128, 256), 2, (n, 256), 2),
        # Omitted (Known Rank)
        (None, None, 2, None, 2),
        # Omitted (Unknown Rank)
        (None, None, -1, None, -1),
    ]
)
def test_qnn_linear_inference(use_bias, dtype, x_shape, w_shape, ndim, expected_shape, expected_ndim):
    bb = relax.BlockBuilder()

    alpha = relax.Var("alpha", relax.TensorStructInfo((), "float32"))
    x = relax.Var("x", _make_sinfo(x_shape, dtype, ndim))
    x_scale = relax.Var("x_scale", relax.TensorStructInfo((), "float32"))
    x_zp = relax.Var("x_zp", relax.TensorStructInfo((), dtype))

    w = relax.Var("w", _make_sinfo(w_shape, dtype, ndim))
    w_scale = relax.Var("w_scale", relax.TensorStructInfo((), "float32"))
    w_zp = relax.Var("w_zp", relax.TensorStructInfo((), dtype))

    y_scale = relax.Var("y_scale", relax.TensorStructInfo((), "float32"))
    y_zp = relax.Var("y_zp", relax.TensorStructInfo((), dtype))

    # Initialize optional bias
    if use_bias:
        if expected_shape is not None:
            # Broadcastable bias shape: (out_features,)
            b_shape = (expected_shape[1],)
            b_ndim = 1
        else:
            b_shape = None
            b_ndim = expected_ndim
        B = relax.Var("B", _make_sinfo(b_shape, "int32", b_ndim))
    else:
        B = None

    params = [alpha, x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp]
    if B is not None:
        params.append(B)

    with bb.function("main", params=params):
        out = relax.op.qnn.linear(
            alpha, x, x_scale, x_zp,
            w, w_scale, w_zp,
            y_scale, y_zp,
            B=B
        )
        bb.emit_func_output(out)

    func = bb.get()["main"]
    _check_tensor_sinfo(func.ret_struct_info, dtype, expected_shape, expected_ndim)

@pytest.mark.parametrize("out_dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "x_shape, x_ndim, qaxis, exp_q_shape, exp_q_ndim, exp_p_shape, exp_p_ndim",
    [
        # Concrete per-axis
        ((10, 20, 30), 3, 1, (10, 20, 30), 3, (20,), 1),
        # Symbolic per-axis
        ((n, c, h), 3, 1, (n, c, h), 3, (c,), 1),
        # Omitted known rank per-axis (scale should fallback to 1D)
        (None, 3, 1, None, 3, None, 1),
        # Omitted unknown rank per-axis (qaxis fallback)
        (None, -1, 1, None, -1, None, 1),
        # Concrete per-tensor
        ((10, 20, 30), 3, None, (10, 20, 30), 3, (), 0),
    ]
)
def test_qnn_dynamic_quantize_inference(
    out_dtype, x_shape, x_ndim, qaxis, exp_q_shape, exp_q_ndim, exp_p_shape, exp_p_ndim
):
    bb = relax.BlockBuilder()

    x = relax.Var("x", _make_sinfo(x_shape, "float32", x_ndim))

    with bb.function("main", params=[x]):
        out = relax.op.qnn.dynamic_quantize(x, qaxis=qaxis, out_dtype=out_dtype)
        bb.emit_func_output(out)

    func = bb.get()["main"]
    ret_sinfo = func.ret_struct_info

    assert isinstance(ret_sinfo, relax.TupleStructInfo)
    assert len(ret_sinfo.fields) == 3

    q_sinfo, scale_sinfo, zp_sinfo = ret_sinfo.fields

    _check_tensor_sinfo(q_sinfo, out_dtype, exp_q_shape, exp_q_ndim)
    _check_tensor_sinfo(scale_sinfo, "float32", exp_p_shape, exp_p_ndim)
    _check_tensor_sinfo(zp_sinfo, out_dtype, exp_p_shape, exp_p_ndim)

@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "x_shape, ndim, expected_shape, expected_ndim",
    [
        # Concrete: (N, C, H, W) with pool_size=(3, 3), strides=(2, 2), padding=(1, 1, 1, 1)
        # H_out = (H_in + 2*pad - pool_size) // stride + 1 = (56 + 2 - 3) // 2 + 1 = 28
        ((1, 64, 56, 56), 4, (1, 64, 28, 28), 4),
        # Symbolic
        ((n, c, h, w), 4, (n, c, (h - 1) // 2 + 1, (w - 1) // 2 + 1), 4),
        # Omitted (Known Rank)
        (None, 4, None, 4),
        # Omitted (Unknown Rank)
        (None, -1, None, -1),
    ]
)
def test_qnn_avg_pool2d_inference(dtype, x_shape, ndim, expected_shape, expected_ndim):
    bb = relax.BlockBuilder()

    x = relax.Var("x", _make_sinfo(x_shape, dtype, ndim))
    x_scale = relax.Var("x_scale", relax.TensorStructInfo((), "float32"))
    x_zp = relax.Var("x_zp", relax.TensorStructInfo((), dtype))

    y_scale = relax.Var("y_scale", relax.TensorStructInfo((), "float32"))
    y_zp = relax.Var("y_zp", relax.TensorStructInfo((), dtype))

    with bb.function("main", params=[x, x_scale, x_zp, y_scale, y_zp]):
        out = relax.op.qnn.avg_pool2d(
            x, x_scale, x_zp,
            y_scale, y_zp,
            pool_size=(3, 3), strides=(2, 2), padding=(1, 1, 1, 1)
        )
        bb.emit_func_output(out)

    func = bb.get()["main"]
    # For avg_pool2d, output dtype is inherited from the input
    _check_tensor_sinfo(func.ret_struct_info, dtype, expected_shape, expected_ndim)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "x_shape, ndim, expected_shape, expected_ndim",
    [
        # Concrete 2D
        ((32, 1000), 2, (32, 1000), 2),
        # Concrete 4D
        ((1, 3, 224, 224), 4, (1, 3, 224, 224), 4),
        # Symbolic
        ((n, c), 2, (n, c), 2),
        # Omitted (Known Rank)
        (None, 2, None, 2),
        # Omitted (Unknown Rank)
        (None, -1, None, -1),
    ]
)
def test_qnn_softmax_inference(dtype, x_shape, ndim, expected_shape, expected_ndim):
    bb = relax.BlockBuilder()

    beta = relax.Var("beta", relax.TensorStructInfo((), "float32"))
    x = relax.Var("x", _make_sinfo(x_shape, dtype, ndim))
    x_scale = relax.Var("x_scale", relax.TensorStructInfo((), "float32"))
    x_zp = relax.Var("x_zp", relax.TensorStructInfo((), dtype))

    y_scale = relax.Var("y_scale", relax.TensorStructInfo((), "float32"))
    y_zp = relax.Var("y_zp", relax.TensorStructInfo((), dtype))

    with bb.function("main", params=[beta, x, x_scale, x_zp, y_scale, y_zp]):
        out = relax.op.qnn.softmax(
            beta, x, x_scale, x_zp,
            y_scale, y_zp,
            axis=-1
        )
        bb.emit_func_output(out)

    func = bb.get()["main"]
    _check_tensor_sinfo(func.ret_struct_info, dtype, expected_shape, expected_ndim)
