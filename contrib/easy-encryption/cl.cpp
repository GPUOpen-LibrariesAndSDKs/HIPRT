#include <stdio.h>
#include <string.h>
#include <string>
#include <iostream>
#include <stdio.h>
#include <ctype.h>
#include <fstream>
#include "encrypt.h"

using namespace std;

string load( const char* path )
{
    FILE* src = fopen(path, "rb");
    string res;

    fseek(src,0,SEEK_END);
    int size = ftell(src);
	rewind( src );
	res.resize(size);
    fread(const_cast<char*>(res.data()), size, 1, src);
    fclose(src);
    return res;
}

void write( const char* path, string& str )
{
# if 1
    FILE* fp = fopen( path, "wb" );
    fwrite( str.c_str(), strlen( str.c_str() ), 1, fp );
    fclose( fp );
#else
    ofstream ofs(path);
    ofs << str;
    ofs.close();    
#endif
}

int main(int argc, char** argv) 
{
//ee64 src.txt dst.txt key 0
#if 1
    std::string key = argv[3];
    int encrypt_flag = atoi(argv[4]);

    string src = load( argv[1] );
    string dst;
    if(encrypt_flag == 0) {
        dst = encrypt(src, key);
    } else {
        dst = decrypt(src, key);
    }

    write( argv[2], dst );
#else
 	std::string msg = argv[1];
 	std::string key = argv[2];
 	int encrypt_flag = atoi(argv[3]);

 	if(encrypt_flag == 0) {
 	    std::cout << encrypt(msg, key) << std::endl;
 	} else {
 	    std::cout << decrypt(msg, key) << std::endl;
 	}
#endif
    return 0;
}
