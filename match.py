class Point:
    __match_args__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

def locate(point):
    match point:
        case Point(0, 0):
            print(f"Point is at origin")
        case Point(x = 1):
            print(f"Point is at 1, {point.y}")    
        case _:
            print(f"Point is somewhere")

def main():
    p = Point(x = 1, y = 5)
    locate(p)

if __name__ ==  "__main__":
    main()
