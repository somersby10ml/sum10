
class Sum10:

    def calc(self, number_array) -> tuple:
        """
        Return the area that sums to 10 by first dragging
        :param number_array:
        :return: x, y, width, height
        """
        rows = len(number_array)
        cols = len(number_array[0])

        cnt = 0
        while True:
            # find x array
            for x in range(cols):
                while True:
                    idx = self.array_find_rows_Value(number_array, x)
                    if idx == -1:
                        break

                    for a in range(idx[0], idx[0] + idx[1]):
                        number_array[a][x] = -1

                    # self.Drag(x, idx[0], 1, idx[1])
                    return x, idx[0], 1, idx[1]
                    # cnt = cnt + 1

            # find y array
            for y in range(rows):
                while True:
                    idx = self.array_find_cols_Value(number_array[y])
                    if idx == -1:
                        break

                    for a in range(idx[0], idx[0] + idx[1]):
                        number_array[y][a] = -1

                    # self.Drag(idx[0], y, idx[1], 1)
                    return idx[0], y, idx[1], 1
                    # cnt = cnt + 1

            if cnt == 0:
                break

        return 0, 0, 0, 0

    def array_find_cols_Value(self, arr):
        """
        가로에서 찾음
        found in horizontal
        :param arr: horizontal array
        :return: x offset,  count
        """
        # 전체를 루프함
        for i in range(len(arr)):
            if arr[i] == -1:
                continue

            # 해당 인덱스부터 10이 되는걸 체크함
            sum = 0
            for x in range(i, len(arr)):

                if arr[x] == -1:
                    continue

                sum += arr[x]
                if sum == 10:
                    return (i, x - i + 1)

                if sum > 10:
                    break
        return -1


    def array_find_rows_Value(self, arr, idx):
        """
        found in vertical
        :param arr: numpy array
        :param idx: vertical index
        :return: y offset,  count
        """
        # 전체를 루프함
        for i in range(len(arr)):
            if arr[i][idx] == -1:
                continue

            # 해당 인덱스부터 10이 되는걸 체크함
            sum = 0
            for y in range(i, len(arr)):

                if arr[y][idx] == -1:
                    continue

                sum += arr[y][idx]
                if sum == 10:
                    return (i, y - i + 1)

                if sum > 10:
                    break
        return -1

    def array_find_square(self):
        # todo
        pass


