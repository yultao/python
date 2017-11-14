'''
Created on Sep 30, 2017

@author: yultao
'''
import random
import urllib.request

################################
def downimg(url):
    x = random.randrange(1,1000)
    filename = str(x) +".jpg"
    
    print(filename,url)
    urllib.request.urlretrieve(url, filename)

    
# downimg("https://yt3.ggpht.com/--n5ELY2uT-U/AAAAAAAAAAI/AAAAAAAAAAA/d9JvaIEpstw/s88-c-k-no-mo-rj-c0xffffff/photo.jpg")

################################
def filewriter(file):
    fw = open(file, "w")
    fw.write("hello file\n")
    fw.write("another linex")
    fw.close()
    
def filereader(file):
    fr = open(file, "r")
    print(fr.read())
    fr.close()
#filewriter("file.out")
filereader("file.out")

################################


