from collections import deque


def deque_test():
    fruits = deque(["oranges", "apples", "bananas"])
    fruits.appendleft("grapes")
    fruits.append("strawberry")

    print(f"Popped left: {fruits.popleft()}")
    print(f"Popped right: {fruits.pop()}")

    print(f"Remaining fruits: {[fruit for fruit in fruits]}")  # fruits is of type deque; this converts it into a list.


def comprehension_test():
    mat = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    tranpose_mat = [[row[i] for row in mat] for i in range(0, 4)]
    print(f"mat: {mat}, transpose: {tranpose_mat}")

    tranpose_alt = zip(*mat)  # zip combines elements from incoming lists; this results in the transpose
    print(f"Alernate transpose computation: {tranpose_mat}")


def sorted_set():
    ids = [999, 9, 9, 1, 77]
    sorted_ids = [id for id in sorted(set(ids))]
    print(f"Sorted ids: {sorted_ids}")


if __name__ == "__main__":
    deque_test()
    comprehension_test()
    sorted_set()
