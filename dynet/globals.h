#ifndef DYNET_GLOBALS_H
#define DYNET_GLOBALS_H

#include <random>
#include "utils.h"

namespace dynet {

class Device;
class NamedTimer;

extern std::mt19937* rndeng;
extern Device* default_device;
extern NamedTimer timer; // debug timing in executors.
extern std::string store_file;
extern OoC::Timer global_timer;
extern std::string schedule_alg;
extern bool blocked;
extern bool opt_mem;
extern int do_pre_malloc;
extern int ooc_autobatch_flag;
extern int block_opt_flag;

} // namespace dynet

#endif
