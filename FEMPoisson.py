f = -1

# базисные ("шляпные") функции в 1D: x или 1-x
hat0 = {"eval": lambda x: x, "nabla": lambda x: 1}
hat1 = {"eval": lambda x: 1 - x, "nabla": lambda x: -1}
hatFunction = [hat0, hat1]


# базисная ("шляпная") функция в 2D = произведение двух базисных функций из 1D
class Basis2D:
    def __init__(self, x0, y0):
        self.xBasis = hatFunction[x0]
        self.yBasis = hatFunction[y0]

    # значение базисной функции в точке x, y
    def evaluation(self, x, y):
        return self.xBasis["eval"](x) * self.yBasis["eval"](y)

    # производная базисной функции в точке x, y (на самом деле оператор набла (Гамильтона)
    # дает сумму двух значений, произведение двух сумм (a+b)*(c+d) = ac + ad + bc + bd,
    # поэтому возвращаем вектор, состоящий из двух элементов суммы
    def nabla(self, x, y):
        return [self.xBasis["nabla"](x) * self.yBasis["eval"](y),
                self.yBasis["nabla"](y) * self.xBasis["eval"](x)]


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


# Mash - класс, сетка.
class Mash:
    def __init__(self, xMin=0.0, xMax=1.0, yMin=0.0, yMax=1.0, stepSize=1.0):
        self._xNum, self._yNum = 0, 0
        self._xMesh, self._yMesh = [], []
        self._xMin, self._xMax, self._yMin, self._yMax = xMin, xMax, yMin, yMax
        self._h = stepSize
        self._points, self._cells = [], []

    def get_coordinates(self, coord_list, cMin, cMax, rounding):
        # находит координаты от мин до макс, округляя
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

        for ind in range(self._yNum):
            self._xMesh.append(x_coordinates)
            self._yMesh.append([y_coordinates[ind]] * self._xNum)

        for i in range(len(self._yMesh)):
            for j in range(self._xNum):
                self._points.append(Point(self._xMesh[i][j], self._yMesh[i][j], i * self._xNum + j))

        for i, point in enumerate(self._points):
            if point.x != self._xMax and point.y != self._yMax:
                self._cells.append(
                    Cell(
                        self._points,
                        self._points[i + 1],
                        self._points[i + self._xNum],
                        self._points[i + 1 + self._xNum]
                    )
                )

    def createSystem(self):
        # # system matrix
        # A = dok_matrix((len(self.dofs), len(self.dofs)), dtype=np.float32)
        # # system right hand side
        F = [0]*len(self._points)

        for cell in self._cells:
           for j, dof_j in enumerate(cell._points):
                    # assemble rhs
                F[dof_j.ind] = ((1/6) ** 2) * f * (self._h ** 2) *\
                    (basisFunction[j].evaluation(0, 0) + 4 * basisFunction[j].evaluation(0, 1/2) +
                     basisFunction[j].evaluation(0, 1) + 4 * basisFunction[j].evaluation(1/2, 0) +
                     16 * basisFunction[j].evaluation(1/2, 1/2) + 4 * basisFunction[j].evaluation(1/2, 1) +
                     basisFunction[j].evaluation(1, 0) + 4 * basisFunction[j].evaluation(1, 1/2) +
                     basisFunction[j].evaluation(1, 1))


mash = Mash(xMax=1.5,stepSize=0.5)
mash.createMesh()
