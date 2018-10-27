def f1():
    cache = (1,2,3)
    return cache
def f2():
    cache1 = f1()
    cache2 = ("X", "Y")
    cache3 = (cache1, cache2)
    return cache3

c1,c2 = f2();
print(c1)

print(c2)

one, two, three = c1
print(one)
print(two)
print(three)

print(f2()[0][0])

