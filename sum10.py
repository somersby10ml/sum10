import numpy

class Sum10:

    def calc(self, number_array) -> tuple:
        """
        Return the area that sums to 10 by first dragging
        :param number_array:
        :return: x, y, width, height
        """
        rows = len(number_array)
        cols = len(number_array[0])

        location = numpy.argwhere(number_array != 0)
        # 위치를 찾고
        for y, x in location:
            # 해당 위치부터 스캔함
            for yy in range(y, rows+1):
                for xx in range(x, cols+1):

                    # 첫번째는 패스
                    if numpy.size(number_array[y:yy, x:xx]) < 1:
                        continue

                    # 10 이 넘으면
                    s = numpy.sum(number_array[y:yy, x:xx])
                    if s > 10:
                        continue

                    # ok
                    if s == 10:
                        number_array[y:yy, x:xx].fill(0)
                        return x, y, xx-x, yy-y

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

                if arr[x] == 0:
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
            if arr[i][idx] == 0:
                continue

            # 해당 인덱스부터 10이 되는걸 체크함
            sum = 0
            for y in range(i, len(arr)):

                if arr[y][idx] == 0:
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


