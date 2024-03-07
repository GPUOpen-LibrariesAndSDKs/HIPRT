f = open("version.txt", "r")
ma = int(f.readline())
mi = int(f.readline())
f.close()

print( ma * 1000 + mi )

f = open("tmp.txt", "w")
f.write( str( ma * 1000 + mi ) )
f.close()