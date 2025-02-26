#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import subprocess
import platform

dir = './'
ekey = ''
defines = {}
replaced = []

# A classic usage is to have this optional temporary env var created during the cmake process (through the 'NO_ENCRYPT' cmake option) before calling this python script.
# If this env var is ON, it overides the encyption and disable it.
no_encrypt = os.getenv('HIPRT_NO_ENCRYPT') == 'ON'

def encrypt(message, key):
    file = open('./tmp.txt',mode='w')
    file.write(message)
    file.close()
    binary = ''
    if platform.system() == 'Windows':
        binary = './contrib/easy-encryption/bin/win/ee64'
    elif platform.system() == 'Darwin':
        binary = './contrib/easy-encryption/bin/macos/ee64'
    else:
        binary = './contrib/easy-encryption/bin/linux/ee64'

    subprocess.check_output([binary, './tmp.txt', './tmp.bin', key, "0"])

    file1 = open('./tmp.bin',mode='r')
    msg = file1.read()
    file1.close()
    return msg
#    return subprocess.check_output(['./adl/contrib/lib/ee/win/ee64', message, key, "0"]).decode('utf8').strip()

def registerDefs(line):
    global defines
    return 0
    # Changed to include leading whitespace, as this was causing issue with defines that contain 'TH_' inside the name.
    if line.find(' TH_') == -1:
        return 0
    keys = line.split(' ',1)
    keys = keys[1].split(' ',1)

    if keys[0] in defines:
      return 0
    if len(keys) >= 2:
      defines[keys[0]] = keys[1]
      for i in range(2, len(keys)):
        defines[keys[0]] += " " + keys[i]
    return 1

def replaceDefines( line ):
    defKeys = defines.keys()
    for key in defKeys:
        if line.find(key) != -1:
            line = line.replace(key, defines[key])
            replaced.append( line )
    return line

def removeLeadingSpace( src ):
    return src.lstrip(' ').rstrip(' ')

def printfile(filename, ans, enablePrint, api):
    fh = open(filename)

    for line in fh.readlines():
        a = line.strip('\r\n')
        a = removeLeadingSpace( a )
        if a.startswith('//'):
            continue
        if a.find('#include') != -1 and (a.find('inl.cl') != -1 or a.find('inl.metal') != -1 or a.find('inl.cu') != -1):
            head, tail = os.path.split(a)
            tail = dir + tail
            tail = tail.replace( '>', '' )
            printfile( tail, ans, 0, api )
        if a.find('#include') != -1 and (api != 'hip'):
            continue
        if a.find('#define') == 0:
            if registerDefs( a ) == 1:
                continue
        if a.find('#undef') == -1:
            a = replaceDefines( a )
        if( ekey != '' ):
          b = a
        else:
          b = ('"'+a.replace("\\", "\\\\").replace("\"", "\\\"").replace("'", "\\'") + '\\n"')
        ans += ''+b+'\n'
    return ans

def stringify(filename, stringname, api):
    print ('static const char* '+stringname+'= \\')
    ans = ''
    ans = printfile( filename, ans, 1, api )
#    print(len(ans))
#    print( ans )
    if( ekey != '' ):
        ans = encrypt( ans, ekey )
#    print( '"'+ans+'";' )
        chars_per_line = 255
        for i in range(0, len(ans), chars_per_line):
            print( '"'+ans[i:i+chars_per_line]+'"\\')
        print(';')
    else:
        print( ans + ';' )

argvs = sys.argv

files = []
if len(argvs) >= 2:
    files.append( argvs[1] )

if len(argvs) >= 3:
    ekey = argvs[2]
    if no_encrypt:
        ekey = ''


#files = ['IntegratorGpuLightSamplerTestKernel.cl','Math.cl']

api = 'cl' if any(name.endswith('cl') for name in files) else 'metal'
if (api == 'metal'):
    api = 'cu' if any(name.endswith('cu') for name in files) else 'metal'

api = 'hip'

for file in files:
    if file.find('Math.')==-1:
        continue
    if file.find('.cl') == -1 and file.find('.cu') == -1 and file.find('.metal') == -1 and file.find('.h') == -1:
        continue
    stringname = file.replace('.cl', '').replace('.cu', '').replace('.metal', '').replace('.h', '')
    stringname = api + '_'+stringname.split('/')[-1]
    stringify( dir+file, stringname, api )
#    print (stringname)

for file in files:
    if file.find('Math.')!=-1:
        continue
    if file.find('.cl') == -1 and file.find('.cu') == -1 and file.find('.metal') == -1 and file.find('.h') == -1:
        continue
    stringname = file.replace('.cl', '').replace('.cu', '').replace('.metal', '').replace('.h', '')
    stringname = api + '_'+stringname.split('/')[-1]
    stringify( dir+file, stringname, api )
#
log = open('tahoePy.log', 'w')
log.write(">> Registerd Defs\n")
defKeys = defines.keys()
for key in defKeys:
    log.write( "  #define {0}   => {1}\n".format(key, defines[key]) )
log.write("\n")
log.write(">> Replaced Defs\n")
for r in replaced:
    log.write( "  %s\n"%r )
log.close()
