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

def _make_qparam_vars(name_prefix, shape, zp_dtype, fallback_ndim=-1):
    """Helper to generate scale and zero-point vars with specific quantization shapes."""
    if shape is None:
        scale_sinfo = relax.TensorStructInfo(None, "float32", ndim=fallback_ndim)
        zp_sinfo = relax.TensorStructInfo(None, zp_dtype, ndim=fallback_ndim)
    else:
        scale_sinfo = relax.TensorStructInfo(shape, "float32")
        zp_sinfo = relax.TensorStructInfo(shape, zp_dtype)

    return relax.Var(f"{name_prefix}_scale", scale_sinfo), relax.Var(f"{name_prefix}_zp", zp_sinfo)

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


# --- TESTS ---

@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "a_shape, a_q_shape, b_shape, b_q_shape, out_q_shape, ndim, expected_shape, expected_ndim",
    [
        # Concrete (Per-tensor)
        ((1, 3, 224, 224), (), (1, 3, 224, 224), (), (), 4, (1, 3, 224, 224), 4),
        # Concrete (Per-axis on channels, dim=1)
        ((1, 3, 224, 224), (1, 3, 1, 1), (1, 3, 224, 224), (1, 3, 1, 1), (1, 3, 1, 1), 4, (1, 3, 224, 224), 4),
        # Concrete (Per-block on spatial dims, e.g., 16x16 blocks -> 224/16 = 14)
        ((1, 3, 224, 224), (1, 3, 14, 14), (1, 3, 224, 224), (1, 3, 14, 14), (), 4, (1, 3, 224, 224), 4),
        # Symbolic Broadcast
        ((n, c, h, w), (), (1, c, 1, 1), (), (), 4, (n, c, h, w), 4),
        # Symbolic (Per-axis on channels)
        ((n, c, h, w), (1, c, 1, 1), (n, c, h, w), (1, c, 1, 1), (), 4, (n, c, h, w), 4),
        # Omitted Shape (Known Rank, Unknown Q-Rank)
        (None, None, None, None, None, 4, None, 4),
        # Omitted Shape (Unknown Rank)
        (None, None, None, None, None, -1, None, -1),
    ]
)
def test_qnn_add_inference(dtype, a_shape, a_q_shape, b_shape, b_q_shape, out_q_shape, ndim, expected_shape, expected_ndim):
    bb = relax.BlockBuilder()

    a = relax.Var("a", _make_sinfo(a_shape, dtype, ndim))
    a_scale, a_zp = _make_qparam_vars("a", a_q_shape, dtype)

    b = relax.Var("b", _make_sinfo(b_shape, dtype, ndim))
    b_scale, b_zp = _make_qparam_vars("b", b_q_shape, dtype)

    c_scale, c_zp = _make_qparam_vars("c", out_q_shape, dtype)

    with bb.function("main", params=[a, a_scale, a_zp, b, b_scale, b_zp, c_scale, c_zp]):
        out = relax.op.qnn.add(a, a_scale, a_zp, b, b_scale, b_zp, c_scale, c_zp)
        bb.emit_func_output(out)

    func = bb.get()["main"]
    _check_tensor_sinfo(func.ret_struct_info, dtype, expected_shape, expected_ndim)


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "x_shape, w_shape, w_q_shape, ndim, expected_shape, expected_ndim",
    [
        # Per-tensor
        ((1, 64, 56, 56), (128, 64, 3, 3), (), 4, (1, 128, 28, 28), 4),
        # Per-axis on weights (out_channels = 128)
        ((1, 64, 56, 56), (128, 64, 3, 3), (128, 1, 1, 1), 4, (1, 128, 28, 28), 4),
        # Per-block on weights (block size 32 on in_channels -> 64/32 = 2)
        ((1, 64, 56, 56), (128, 64, 3, 3), (128, 2, 1, 1), 4, (1, 128, 28, 28), 4),
        # Symbolic: out_dim = (in_dim + pad - kernel) // stride + 1
        ((n, 64, h, w), (128, 64, 3, 3), (), 4, (n, 128, (h - 1) // 2 + 1, (w - 1) // 2 + 1), 4),
        # Omitted Shapes
        (None, None, None, 4, None, 4),
        (None, None, None, -1, None, -1),
    ]
)
def test_qnn_conv2d_inference(use_bias, dtype, x_shape, w_shape, w_q_shape, ndim, expected_shape, expected_ndim):
    bb = relax.BlockBuilder()

    x = relax.Var("x", _make_sinfo(x_shape, dtype, ndim))
    x_scale, x_zp = _make_qparam_vars("x", (), dtype)  # Acts per-tensor

    w = relax.Var("w", _make_sinfo(w_shape, dtype, ndim))
    w_scale, w_zp = _make_qparam_vars("w", w_q_shape, dtype) # Parameterized for axis/block

    y_scale, y_zp = _make_qparam_vars("y", (), dtype)

    if use_bias:
        if expected_shape is not None:
            b_shape = (expected_shape[1], 1, 1)
            b_ndim = 3
        else:
            b_shape = None
            b_ndim = expected_ndim
        B = relax.Var("B", _make_sinfo(b_shape, "int32", b_ndim))
    else:
        B = None

    params = [x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp]
    if B is not None:
        params.append(B)

    with bb.function("main", params=params):
        out = relax.op.qnn.conv2d(
            x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp,
            B=B, strides=(2, 2), padding=(1, 1, 1, 1)
        )
        bb.emit_func_output(out)

    func = bb.get()["main"]
    _check_tensor_sinfo(func.ret_struct_info, dtype, expected_shape, expected_ndim)


@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "x_shape, x_q_shape, w_shape, w_q_shape, ndim, expected_shape, expected_ndim",
    [
        # Per-tensor
        ((32, 128), (), (128, 256), (), 2, (32, 256), 2),
        # Per-axis on weights (out_features = 256)
        ((32, 128), (), (128, 256), (1, 256), 2, (32, 256), 2),
        # Per-block on weights (block size 32 over K dimension -> 128/32 = 4)
        ((32, 128), (), (128, 256), (4, 256), 2, (32, 256), 2),
        # Per-block on activations (block size 32 over K dim)
        ((32, 128), (32, 4), (128, 256), (4, 256), 2, (32, 256), 2),
        # Symbolic
        ((n, 128), (), (128, 256), (), 2, (n, 256), 2),
        # Omitted
        (None, None, None, None, 2, None, 2),
        (None, None, None, None, -1, None, -1),
    ]
)
def test_qnn_linear_inference(use_bias, dtype, x_shape, x_q_shape, w_shape, w_q_shape, ndim, expected_shape, expected_ndim):
    bb = relax.BlockBuilder()

    x = relax.Var("x", _make_sinfo(x_shape, dtype, ndim))
    x_scale, x_zp = _make_qparam_vars("x", x_q_shape, dtype)

    w = relax.Var("w", _make_sinfo(w_shape, dtype, ndim))
    w_scale, w_zp = _make_qparam_vars("w", w_q_shape, dtype)

    y_scale, y_zp = _make_qparam_vars("y", (), dtype)

    if use_bias:
        if expected_shape is not None:
            b_shape = (expected_shape[1],)
            b_ndim = 1
        else:
            b_shape = None
            b_ndim = expected_ndim
        B = relax.Var("B", _make_sinfo(b_shape, "int32", b_ndim))
    else:
        B = None

    params = [x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp]
    if B is not None:
        params.append(B)

    with bb.function("main", params=params):
        out = relax.op.qnn.linear(
            x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, B=B
        )
        bb.emit_func_output(out)

    func = bb.get()["main"]
    _check_tensor_sinfo(func.ret_struct_info, dtype, expected_shape, expected_ndim)


@pytest.mark.parametrize("out_dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "x_shape, x_ndim, axis, exp_q_shape, exp_q_ndim, exp_p_shape, exp_p_ndim",
    [
        # Concrete per-axis
        ((10, 20, 30), 3, 1, (10, 20, 30), 3, (20,), 1),
        # Symbolic per-axis
        ((n, c, h), 3, 1, (n, c, h), 3, (c,), 1),
        # Omitted known rank per-axis
        (None, 3, 1, None, 3, None, 1),
        # Omitted unknown rank per-axis
        (None, -1, 1, None, -1, None, 1),
        # Concrete per-tensor
        ((10, 20, 30), 3, None, (10, 20, 30), 3, (), 0),
    ]
)
def test_qnn_dynamic_quantize_inference(out_dtype, x_shape, x_ndim, axis, exp_q_shape, exp_q_ndim, exp_p_shape, exp_p_ndim):
    bb = relax.BlockBuilder()

    x = relax.Var("x", _make_sinfo(x_shape, "float32", x_ndim))

    with bb.function("main", params=[x]):
        out = relax.op.qnn.dynamic_quantize(x, axis=axis, out_dtype=out_dtype)
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
    "x_shape, x_q_shape, ndim, expected_shape, expected_ndim",
    [
        # Concrete (Per-tensor)
        ((1, 64, 56, 56), (), 4, (1, 64, 28, 28), 4),
        # Concrete (Per-axis on channels)
        ((1, 64, 56, 56), (1, 64, 1, 1), 4, (1, 64, 28, 28), 4),
        # Symbolic
        ((n, c, h, w), (), 4, (n, c, (h - 1) // 2 + 1, (w - 1) // 2 + 1), 4),
        # Omitted
        (None, None, 4, None, 4),
        (None, None, -1, None, -1),
    ]
)
def test_qnn_avg_pool2d_inference(dtype, x_shape, x_q_shape, ndim, expected_shape, expected_ndim):
    bb = relax.BlockBuilder()

    x = relax.Var("x", _make_sinfo(x_shape, dtype, ndim))
    x_scale, x_zp = _make_qparam_vars("x", x_q_shape, dtype)

    y_scale, y_zp = _make_qparam_vars("y", (), dtype)

    with bb.function("main", params=[x, x_scale, x_zp, y_scale, y_zp]):
        out = relax.op.qnn.avg_pool2d(
            x, x_scale, x_zp, y_scale, y_zp,
            pool_size=(3, 3), strides=(2, 2), padding=(1, 1, 1, 1)
        )
        bb.emit_func_output(out)

    func = bb.get()["main"]
    _check_tensor_sinfo(func.ret_struct_info, dtype, expected_shape, expected_ndim)


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "x_shape, x_q_shape, ndim, expected_shape, expected_ndim",
    [
        # Concrete 2D (Per-tensor)
        ((32, 1000), (), 2, (32, 1000), 2),
        # Concrete 2D (Per-axis)
        ((32, 1000), (32, 1), 2, (32, 1000), 2),
        # Concrete 4D (Per-tensor)
        ((1, 3, 224, 224), (), 4, (1, 3, 224, 224), 4),
        # Symbolic
        ((n, c), (), 2, (n, c), 2),
        # Omitted
        (None, None, 2, None, 2),
        (None, None, -1, None, -1),
    ]
)
def test_qnn_softmax_inference(dtype, x_shape, x_q_shape, ndim, expected_shape, expected_ndim):
    bb = relax.BlockBuilder()

    x = relax.Var("x", _make_sinfo(x_shape, dtype, ndim))
    x_scale, x_zp = _make_qparam_vars("x", x_q_shape, dtype)

    y_scale, y_zp = _make_qparam_vars("y", (), dtype)

    with bb.function("main", params=[x, x_scale, x_zp, y_scale, y_zp]):
        out = relax.op.qnn.softmax(x, x_scale, x_zp, y_scale, y_zp, axis=-1)
        bb.emit_func_output(out)

    func = bb.get()["main"]
    _check_tensor_sinfo(func.ret_struct_info, dtype, expected_shape, expected_ndim)
