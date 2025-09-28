import numpy as np

def main():
    mat1 = np.array([[2, 3, 4],
                     [5, 6, 7],
                     [8, 9, 10]], np.float32)
    np.identity(3, np.float32)
    print(f"mat1: {mat1}")

if __name__ ==  "__main__":
    main()
