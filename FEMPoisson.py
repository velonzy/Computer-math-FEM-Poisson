from matplotlib import pyplot as plt

# левая часть дифференциального уравнения, уравнения Пуассона
f = -1

# базисные ("шляпные") функции в 1D: x или 1-x (если по оси OY, то y и 1-y)
hat0 = {"eval": lambda x: x, "nabla": lambda x: 1}
hat1 = {"eval": lambda x: 1 - x, "nabla": lambda x: -1}
hatFunction = [hat0, hat1]


# базисная ("шляпная") функция в 2D = произведение двух базисных функций из 1D
class Basis2D:
    def __init__(self, x0, y0):
        self.xBasis = hatFunction[x0]
        self.yBasis = hatFunction[y0]

    def evaluation(self, x, y):
        # значение базисной функции в точке x, y
        return self.xBasis["eval"](x) * self.yBasis["eval"](y)

    def nabla(self, x, y):
        # производная базисной функции в точке x, y (на самом деле оператор набла (Гамильтона)
        # дает сумму двух значений, произведение двух сумм (a+b)*(c+d) = ac + ad + bc + bd,
        # поэтому возвращаем вектор, состоящий из двух элементов суммы
        return [self.xBasis["nabla"](x) * self.yBasis["eval"](y),
                self.yBasis["nabla"](y) * self.xBasis["eval"](x)]

    def scalar(self, x_self, y_self, other, x_other, y_other):
        # скалярное произведение двух базисных функций
        return (self.nabla(x_self, y_self)[0] * other.nabla(x_other, y_other)[0] +
                self.nabla(x_self, y_self)[1] * other.nabla(x_other, y_other)[1])


# линейная базисная функция в 2D: xy, (1-x)y, x(1-y), (1-x)(1-y)
basisFunction = {0: Basis2D(0, 0), 1: Basis2D(1, 0), 2: Basis2D(0, 1), 3: Basis2D(1, 1)}


# Point - класс, хранит точку сетки (одна точка и ее координаты),
# а также ее индекс в сетке
# (порядок - 1, 2, ... 22 (до (n+1)^2, n - количество КЭ))
class Point:
    def __init__(self, x, y, ind):
        self.x, self.y, self.index = x, y, ind


# Cell - класс, хранит 4 точки клетки: левая нижняя - главная. Хранится в виде списка
class Cell:
    def __init__(self, point, right_point, up_point, up_right_point):
        self.points = [point, right_point, up_point, up_right_point]


# Mesh - класс, сетка.
class Mesh:
    def __init__(self, xMin=0.0, xMax=1.0, yMin=0.0, yMax=1.0, stepSize=0.5):
        self._xNum, self._yNum = 0, 0
        self._xMesh, self._yMesh = [], []
        self._xMin, self._xMax, self._yMin, self._yMax = xMin, xMax, yMin, yMax
        self._h = stepSize
        self._points, self._cells = [], []

    def get_coordinates(self, coord_list, cMin, cMax, rounding):
        # находит координаты от мин до макс, округляя, делит сетку по одной координатной оси с шагом h
        ind = cMin
        while ind < cMax:
            coord_list.append(round(ind, rounding))
            ind += self._h
        if coord_list[len(coord_list) - 1] < cMax:
            coord_list.append(cMax)

    def createMesh(self):
        # решаем проблему точности, находим количество знаков после запятой в шаге,
        # округляем до этого количества
        h_tmp = self._h
        sights = 0
        while h_tmp % 10 > 0:
            h_tmp *= 10
            sights += 1

        # нашли координаты по оси X с заданным шагом
        x_coordinates = []
        self.get_coordinates(x_coordinates, self._xMin, self._xMax, sights)
        # количество точек по оси Х
        self._xNum = len(x_coordinates)
        # нашли координаты по оси У с заданным шагом
        y_coordinates = []
        self.get_coordinates(y_coordinates, self._yMin, self._yMax, sights)
        # количество точек по оси Y
        self._yNum = len(y_coordinates)

        # задает сетку х-координат и у-координат (сетка размером х*у),
        # где значение это х-координаты для соответствующих х и у
        for ind in range(self._yNum):
            self._xMesh.append(x_coordinates)
            self._yMesh.append([y_coordinates[ind]] * self._xNum)

        # заполняем список точек
        for i in range(len(self._yMesh)):
            for j in range(self._xNum):
                self._points.append(Point(self._xMesh[i][j], self._yMesh[i][j], i * self._xNum + j))

        # заполняем список клеток (каждая состоит из 4 точек)
        for i, point in enumerate(self._points):
            if point.x != self._xMax and point.y != self._yMax:
                self._cells.append(
                    Cell(
                        point,
                        self._points[i + 1],
                        self._points[i + self._xNum],
                        self._points[i + 1 + self._xNum]
                    )
                )

    def createSystem(self):
        # создает СЛАУ, которую в последствии решаем с помощью LUP-разложения

        # матрица A (двумерная)
        a = [[0 for _ in range(len(self._points))] for _ in range(len(self._points))]
        # матрица F (одномерная)
        _f = [0] * len(self._points)

        # считаем матрицу A и F, причем вместо нахождения значения каждого элемента матрицы с помощью базиса конкретного
        # КЭ, делаем переход к master элементу
        # проходим по всем клеткам (конечным элементам) сетки
        for cell in self._cells:
            # также в каждом конечном элементе проходим по всем его 4‑м точкам
            for j, point_j in enumerate(cell.points):
                # считаем значение Fj для каждой точки КЭ, используем правило Симпсона
                _f[point_j.index] += (((1 / 6) ** 2) * f * (self._h ** 2) *
                                      (basisFunction[j].evaluation(0, 0) + 4 * basisFunction[j].evaluation(0, 1 / 2) +
                                       basisFunction[j].evaluation(0, 1) + 4 * basisFunction[j].evaluation(1 / 2, 0) +
                                       16 * basisFunction[j].evaluation(1 / 2, 1 / 2) +
                                       4 * basisFunction[j].evaluation(1 / 2, 1) + basisFunction[j].evaluation(1, 0) +
                                       4 * basisFunction[j].evaluation(1, 1 / 2) + basisFunction[j].evaluation(1, 1)))
                for i, point_i in enumerate(cell.points):
                    # находим Aij также с помощью правила Симпсона, проходим еще раз по всем точкам одного КЭ,
                    # так как Aij ищется как интеграл от произведения двух базисных функций от разных КЭ
                    a[point_i.index][point_j.index] += (
                            ((1 / 6) ** 2) * (basisFunction[i].scalar(0, 0, basisFunction[j], 0, 0) +
                                              4 * basisFunction[i].scalar(0, 1 / 2, basisFunction[j], 0, 1 / 2) +
                                              basisFunction[i].scalar(0, 1, basisFunction[j], 0, 1) +
                                              4 * basisFunction[i].scalar(1 / 2, 0, basisFunction[j], 1 / 2, 0) +
                                              16 * basisFunction[i].scalar(1 / 2, 1 / 2, basisFunction[j], 1 / 2,
                                                                           1 / 2) +
                                              4 * basisFunction[i].scalar(1 / 2, 1, basisFunction[j], 1 / 2, 1) +
                                              basisFunction[i].scalar(1, 0, basisFunction[j], 1, 0) +
                                              4 * basisFunction[i].scalar(1, 1 / 2, basisFunction[j], 1, 1 / 2) +
                                              basisFunction[i].scalar(1, 1, basisFunction[j], 1, 1)))

        # добавили граничные условия Дирихле, точки, которые на границе = 0, а на главной диагонали = 1 в матрице A,
        # в матрице F также точки с границы равны 0
        for point in self._points:
            if point.x in [self._xMin, self._xMax] or point.y in [self._yMin, self._yMax]:
                for j in range(len(self._points)):
                    a[point.index][j] = 0.0
                a[point.index][point.index] = 1.0
                _f[point.index] = 0.0

        return a, _f

    def plotSolution(self, solution):
        # 2D рисунок решенного уравнения методом КЭ
        plt.contourf(
            self._xMesh,
            self._yMesh,
            solution,
            20,
            cmap="viridis",
        )
        plt.colorbar()
        plt.show()

    def solution_in_2D(self, solution):
        # преобразует вектор решения в матрицу 2D, где каждое решение располагается над своей точкой конечного элемента,
        # нужно для отрисовки
        matrix_solution = [[0 for _ in range(self._xNum)] for _ in range(self._yNum)]
        for i in range(self._yNum):
            for j in range(self._xNum):
                matrix_solution[i][j] = solution[i * self._xNum + j]
        return matrix_solution


def lup_decomposition(a):
    # LUP разложение для матрицы a
    n = len(a)

    # пустые матрицы l и u
    l = [[0.0] * n for i in range(n)]
    u = [[0.0] * n for i in range(n)]

    # матрица перестановок p, в начале на главной диагонали 1
    p = [[float(i == j) for i in range(n)] for j in range(n)]
    # pa - матрица, является произведением матрицы p на a, по сути это сама матрица a
    pa = multiply_matrix(p, a)

    # само LUP-разложение
    for j in range(n):
        # в каждом столбце ставим максимальный по модулю элемент на главную диагональ
        max_elem = a[j][j]
        max_ind = (j, j)
        # находим максимальный
        for k in range(j, n):
            if abs(a[k][j]) > max_elem:
                max_elem = abs(a[k][j])
                max_ind = (k, j)
        # делаем перестановки в матрице pa, p, u и l
        pa[j], pa[max_ind[0]] = pa[max_ind[0]], pa[j]
        p[j], p[max_ind[0]] = p[max_ind[0]], p[j]
        u[j], u[max_ind[0]] = u[max_ind[0]], u[j]
        l[j], l[max_ind[0]] = l[max_ind[0]], l[j]

        # Все значения на главной диагонали матрицы l равны 1
        l[j][j] = 1.0

        for i in range(j + 1):
            # вычисляем значения элементов матриц u и l на текущем шаге
            s1 = sum(u[k][j] * l[i][k] for k in range(i))
            u[i][j] = pa[i][j] - s1
        for i in range(j, n):
            s2 = sum(u[k][j] * l[i][k] for k in range(j))
            l[i][j] = (pa[i][j] - s2) / u[j][j]

    # возвращаем матрицы p - матрица перестановок, l - нижне треугольная матрица с единицами на главной диагонали,
    # u - верхне треугольная матрица
    return p, l, u


def Ly_b(l, b):
    # решает систему ly = b, находит значения вектора y, где l - нижне треугольная матрица с единицами на
    # главной диагонали, на то, что размерности вектора и матрицы подходят проверка не выполняется, так как функция
    # писалась для нашей задачи, то размерности подходят
    y = [0.0 for i in range(len(b))]

    # yi = (bi - sum(lij * yj)) / lii, проходим по матрице l сверху вниз
    for i in range(len(l)):
        summa = sum(l[i][j] * y[j] for j in (range(i)))
        y[i] = (b[i] - summa) / l[i][i]
    return y


def Ux_y(u, y):
    # решает систему ux = y, находит значения вектора x, u - верхне треугольная матрица
    x = [0.0 for i in range(len(y))]

    # xi = (yi - sum(uij * xj)) / uii, проходим по матрице u снизу вверх
    for i in range(len(u) - 1, -1, -1):
        summa = sum(u[i][j] * x[j] for j in range(len(u) - 1, i, -1))
        x[i] = (y[i] - summa) / u[i][i]
    return x


def multiply_matrix(m, n):
    # произведение двух квадратных матриц m и n одинакового размера
    size = len(m)
    result = [[0 for _ in range(len(n))] for _ in range(size)]

    for i in range(size):
        for j in range(len(n)):
            summa = 0
            for k in range(size):
                summa += m[i][k] * n[k][j]
            result[i][j] = summa

    return result


def multiply_matrix_and_vector(m, vec):
    # произведение матрицы m и вектора vec, размеры подходящие
    size = len(m)
    result = [0 for _ in range(len(vec))]

    for i in range(size):
        summa = 0
        for j in range(len(vec)):
            summa += m[i][j] * vec[j]
        result[i] = summa

    return result


if __name__ == '__main__':
    # создаем сетку и систему ЛАУ
    mesh = Mesh()
    mesh.createMesh()
    a, _f = mesh.createSystem()

    # делаем lup-разложение
    p, l, u = lup_decomposition(a)

    # теперь pa = lu
    # решим систему ax = y, домножим слева на p обе части -> pax = py, y = _f, b = p*_f
    # lux = b, ux = y, найдем решение для ly = b, подставим в ux = y, найдем x - искомый вектор
    b = multiply_matrix_and_vector(p, _f)
    y = Ly_b(l, b)
    solution = Ux_y(u, y)

    # найденное решение сделаем удобным для отрисовки в 2D и нарисуем, итоговый рисунок - результат работы программы
    sol = mesh.solution_in_2D(solution)
    mesh.plotSolution(sol)
