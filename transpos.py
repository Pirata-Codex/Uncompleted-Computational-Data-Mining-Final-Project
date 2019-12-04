def transpose(m):
    # m = array([
    #     [11, 22, 33],
    #     [44, 55, 66],
    #     [77, 88, 99]])
    for row in m:
        print(row)
    rez = [[m[j][i] for j in range(len(m))] for i in range(len(m[0]))]
    print("\n")
    for row in rez:
        print(row)