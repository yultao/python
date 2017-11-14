import another
import random
'''
Created on Sep 27, 2017

@author: yultao
'''


if __name__ == '__main__':
    pass

a=101
d=4
print(a//d)
print(a - a//d * d)

############################
def fun(a="default a", b="default b"):
    print("fun",a,b)
    return 1
    
print(fun(b=2))
############################

def anyargs(*args):
    print("anyargs",args)
    for a in args:
        print(a)

anyargs(1,23)

# variable args ###########################

#list
alist=[1,2,3] 

def strange(a, b, c):
    print(a,b,c)
strange(alist[0], alist[1], alist[2])
strange(*alist)

# set ###########################


aset = {"a", "b", "c", "c"}
print(aset)
for a in aset:
    print(a)

# dictionary ###########################
adic = {"f":"female", "m":"male"}
print(adic)
for a in adic:
    print(a,adic[a])
#or
for a,b in adic.items():
    print(a,b)
    
    
###############################


another.whoami()

print(random.randrange(0,100))
