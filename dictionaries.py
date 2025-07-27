def dictionary_test():
    # Approach 1: List of tuples converted to a dictionaty using dict()
    dictionary = dict([
        (0, "Siddharth"),
        (1, "Nidhi"),
        (2, "Vikram"),
        (0, "SiddharthS") # Overwrites the first entry
    ])

    # Approach 2: Initialization using ":"
    # dictionary = {
    #     0 : "Siddharth",
    #     1 : "Nidhi",
    #     2 : "Vikram",
    #     0 : "SiddharthS" # Overwrites the first entry
    # }
    
    for k,v in dictionary.items(): # .items is required
        print(f"key: {k}, value: {v}")

    print(2 in dictionary)

    for i,item in enumerate(dictionary.keys()): # .items is required
        print(f"index: {i}, item: {item}")

if __name__ == "__main__":
    dictionary_test()