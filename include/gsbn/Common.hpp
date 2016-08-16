#ifndef __GSBN_COMMON_HPP__
#define __GSBN_COMMON_HPP__

#include "util/easylogging++.h"
#include "gsbn/proto/gsbn.pb.h"
#include <iomanip>
#include <string>
#include <vector>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

namespace gsbn{

#define __NOT_IMPLEMENTED__ LOG(FATAL) << "Function hasn't been implemented";

}

#endif //__GSBN_COMMON_HPP__
