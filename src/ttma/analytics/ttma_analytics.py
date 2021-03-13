def fat_list():
    l = list()
    for r in range(10000000):
        l.append(r)
    fat = list()
    for r in range(10000000):
        fat.append(l)
    fat2 = fat
    return fat2