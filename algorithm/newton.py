import math


# ================================================
# Функция ввода числа с проверкой
# ================================================
def get_float_input(prompt, forbidden_values=None, non_zero=False, tol=1e-6):
    while True:
        try:
            value = float(input(prompt))
            if non_zero and abs(value) < tol:
                print("Ошибка: значение не может быть 0!")
                continue
            if forbidden_values and any(
                abs(value - fv) < tol for fv in forbidden_values
            ):
                print(f"Ошибка: значение не может быть одним из {forbidden_values}")
                continue
            return value
        except ValueError:
            print("Ошибка: нужно ввести число!")


# ================================================
# Функции и производные
# ================================================
def f(x):
    return math.sqrt(x**2 + 1) * (3 + x)


def g(x):
    return 5 / (x**2 - 7)


def df(x):
    return x * (3 + x) / math.sqrt(x**2 + 1) + math.sqrt(x**2 + 1)


def dg(x):
    return -10 * x / (x**2 - 7) ** 2


def F(x):
    return f(x) - g(x)


def dF(x):
    return df(x) - dg(x)


# ================================================
# Метод Ньютона
# ================================================
def newton(F, dF, x0, tol=1e-6, max_iter=1000, asymptotes=None, asym_tol=1e-6):
    x = x0
    for _ in range(max_iter):
        if asymptotes:
            for a in asymptotes:
                if abs(x - a) < asym_tol:
                    x += math.copysign(asym_tol, x0)
        dFx = dF(x)
        if abs(dFx) < 1e-12:
            raise ZeroDivisionError(f"Производная слишком мала в x={x}")
        x_new = x - F(x) / dFx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise RuntimeError("Метод Ньютона не сошёлся за max_iter итераций")


# ================================================
# Поиск корня шагами h
# ================================================
def find_first_root(
    f, g, x0, h, tol=1e-6, max_steps=10_000_000, asymptotes=None, max_diff=1e-3
):
    F_prev, x_prev = f(x0) - g(x0), x0
    for _ in range(max_steps):
        x_next = x_prev + h
        if asymptotes and any((x_prev - a) * (x_next - a) < 0 for a in asymptotes):
            x_prev += math.copysign(tol, h)
            F_prev = f(x_prev) - g(x_prev)
            continue
        F_next = f(x_next) - g(x_next)
        if F_prev * F_next < 0:
            x_root = newton(
                F, dF, x_prev, tol=tol, max_iter=1000, asymptotes=asymptotes
            )
            if abs(f(x_root) - g(x_root)) <= max_diff:
                return x_root
        x_prev, F_prev = x_next, F_next
    raise RuntimeError("Не удалось найти корень за max_steps шагов")


# ================================================
# Основная программа
# ================================================
if __name__ == "__main__":
    forbidden_x0 = [math.sqrt(7), -math.sqrt(7)]
    x0 = get_float_input("x0: ", forbidden_values=forbidden_x0)
    h = get_float_input("h: ", non_zero=True)

    tol = 1e-6
    asymptotes = [math.sqrt(7), -math.sqrt(7)]

    try:
        root = find_first_root(f, g, x0, h, tol=tol, asymptotes=asymptotes)
        print(
            f"Найденный корень: x ≈ {root:.6f}, f(x) ≈ {f(root):.6f}, g(x) ≈ {g(root):.6f}"
        )
    except (RuntimeError, ZeroDivisionError, ValueError) as e:
        print("Ошибка:", e)
