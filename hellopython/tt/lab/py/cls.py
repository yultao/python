class Enemy:
    a = 1
    
    def __init__(self):
        print(self)
        self.a=2
        self.b = 4
    def other(self):
        
        print("other") 
        print(self.a)
        print(self.b)
e = Enemy()
# e.__init__()
e.other()
print(e.a)
print(e.b)


class Parent:
    def lastname(self):
        print("Tao")
class Parent2:
    def lastname(self):
        print("Xu")
class Child(Parent, Parent2):
    def firstname(self):
        print("George")
        
c = Child()
c.firstname()
c.lastname()