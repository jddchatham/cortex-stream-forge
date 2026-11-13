#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <map>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
namespace diag106 {
inline std::map<std::string,long> memory(){
 std::ifstream f("/proc/self/smaps_rollup");std::string line,key;long v;std::map<std::string,long> r;
 while(std::getline(f,line)){std::istringstream s(line);if(s>>key>>v)r[key]=v;}
 return r;
}
inline void sample(const char*phase,size_t weight,size_t cpu,size_t workspace,size_t metadata,size_t metadata_used,size_t host,size_t source,size_t live,int tile=-1){
 auto m=memory();rusage r{};getrusage(RUSAGE_SELF,&r);
 std::cout<<"{\"phase\":\""<<phase<<"\",\"tile\":"<<tile<<",\"weight_bytes\":"<<weight<<",\"cpu_tensor_buffer_bytes\":"<<cpu<<",\"ggml_workspace_bytes\":"<<workspace<<",\"metadata_capacity_bytes\":"<<metadata<<",\"metadata_used_bytes\":"<<metadata_used<<",\"host_activation_bytes\":"<<host<<",\"source_row_bytes\":"<<source<<",\"live_owned_resources\":"<<live<<",\"rss_kib\":"<<m["Rss:"]<<",\"pss_kib\":"<<m["Pss:"]<<",\"anonymous_kib\":"<<m["Anonymous:"]<<",\"pss_anon_kib\":"<<m["Pss_Anon:"]<<",\"pss_file_kib\":"<<m["Pss_File:"]<<",\"private_clean_kib\":"<<m["Private_Clean:"]<<",\"private_dirty_kib\":"<<m["Private_Dirty:"]<<",\"shared_clean_kib\":"<<m["Shared_Clean:"]<<",\"swap_kib\":"<<m["Swap:"]<<",\"peak_rss_kib\":"<<r.ru_maxrss<<",\"minor_faults\":"<<r.ru_minflt<<",\"major_faults\":"<<r.ru_majflt<<"}\n"<<std::flush;
}
}
