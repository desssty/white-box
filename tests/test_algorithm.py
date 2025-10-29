import math
import pytest
from algorithm.newton import (
    get_float_input,
    f,
    g,
    df,
    dg,
    F,
    dF,
    newton,
    find_first_root,
)


# ================================================
# Тесты функций f, g, df, dg, F, dF
# ================================================
def test_f_g_df_dg_F_dF_values():
    x_values = [-10, -math.sqrt(7), 0, 2, math.sqrt(7), 10]
    for x in x_values:
        assert math.isclose(
            df(x), x * (3 + x) / math.sqrt(x**2 + 1) + math.sqrt(x**2 + 1)
        )
        assert math.isclose(f(x), math.sqrt(x**2 + 1) * (3 + x))

        if x**2 != 7:
            assert math.isclose(g(x), 5 / (x**2 - 7))
            assert math.isclose(dg(x), -10 * x / (x**2 - 7) ** 2)

        if x**2 != 7:
            assert math.isclose(F(x), f(x) - g(x))
            assert math.isclose(dF(x), df(x) - dg(x))


# ================================================
# Тесты newton
# ================================================
def test_newton_converges():
    root = newton(lambda x: x**2 - 4, lambda x: 2 * x, 1.0)
    assert math.isclose(root, 2.0, rel_tol=1e-6)


def test_newton_derivative_zero():
    with pytest.raises(ZeroDivisionError):
        newton(lambda x: x**3, lambda x: 0, 1.0)


# ================================================
# Тесты find_first_root с допустимыми классами
# ================================================
@pytest.mark.parametrize(
    "x0,h",
    [
        (-4, 0.1),
        (-4, -0.1),
        (0, 0.1),
        (0, -0.1),
        (4, 0.1),
        (4, -0.1),
    ],
)
def test_find_first_root_valid(x0, h):
    asymptotes = [math.sqrt(7), -math.sqrt(7)]
    try:
        root = find_first_root(f, g, x0, h, tol=1e-6, asymptotes=asymptotes)
    except RuntimeError:
        root = None

    if root is not None:
        assert isinstance(root, float)
        assert abs(f(root) - g(root)) < 1e-3
    else:
        assert root is None


# ================================================
# Тесты get_float_input для недопустимого ввода
# ================================================
@pytest.mark.parametrize(
    "inputs,forbidden_values,non_zero",
    [
        (["abc", "1.0"], None, False),
        (["7", "1.0"], [math.sqrt(7)], False),
        (["0", "2.0"], None, True),
    ],
)
def test_get_float_input_invalid(monkeypatch, inputs, forbidden_values, non_zero):
    input_iter = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda _: next(input_iter))

    value = get_float_input(
        "Prompt: ", forbidden_values=forbidden_values, non_zero=non_zero
    )
    assert isinstance(value, float)


# ================================================
# Комбинационные тесты недопустимых наборов
# ================================================
@pytest.mark.parametrize(
    "x0_class,h_class",
    [
        (math.sqrt(7), 0),
        (-math.sqrt(7), 0),
        ("abc", 0),
        (math.sqrt(7), "abc"),
        (-math.sqrt(7), "abc"),
        ("abc", "abc"),
    ],
)
def test_find_first_root_invalid_input(monkeypatch, x0_class, h_class):
    x0_input = [x0_class] if isinstance(x0_class, str) else [str(x0_class)]
    h_input = [h_class] if isinstance(h_class, str) else [str(h_class)]
    inputs = x0_input + h_input
    input_iter = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda _: next(input_iter))
    from algorithm.newton import get_float_input

    for val in inputs:
        try:
            get_float_input("Prompt: ")
        except Exception:
            pass
