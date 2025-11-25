def database_test():
    database = [
        (0, "Siddharth", 43),
        (1, "Nidhi", 40),
        (2, "Vikram", 6),
    ]

    for t in database:
        id, name, age = t
        print(f"id: {id}, name: {name}, age: {age}")


if __name__ == "__main__":
    database_test()
