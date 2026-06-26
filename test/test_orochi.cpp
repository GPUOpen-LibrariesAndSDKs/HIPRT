#include "contrib/Orochi/Orochi/Orochi.h"
#include <iostream>

int main() {
    int oroErr = oroInitialize(ORO_API_AUTOMATIC, 0);
    std::cout << "Orochi init: " << oroErr << std::endl;
    return 0;
}
