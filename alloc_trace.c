#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

static __thread int busy;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t);
static int (*real_munmap)(void *, size_t);
static void *(*real_malloc)(size_t);
static void (*real_free)(void *);
static void resolve(void) {
    if (!real_mmap) real_mmap = dlsym(RTLD_NEXT, "mmap");
    if (!real_munmap) real_munmap = dlsym(RTLD_NEXT, "munmap");
    if (!real_malloc) real_malloc = dlsym(RTLD_NEXT, "malloc");
    if (!real_free) real_free = dlsym(RTLD_NEXT, "free");
}
static void trace(const char * kind, void * p, size_t n) {
    if (busy) return; busy = 1;
    dprintf(STDERR_FILENO, "ALLOC_TRACE kind=%s ptr=%p size=%zu\n", kind, p, n);
    void * bt[24]; int nb = backtrace(bt, 24); backtrace_symbols_fd(bt, nb, STDERR_FILENO);
    busy = 0;
}
void * mmap(void * a, size_t n, int prot, int flags, int fd, off_t off) {
    resolve(); void * p = real_mmap(a,n,prot,flags,fd,off);
    if ((flags & MAP_ANONYMOUS) && n >= 16*1024*1024) trace("mmap_anon", p, n);
    return p;
}
int munmap(void * a, size_t n) { resolve(); if (n >= 16*1024*1024) trace("munmap", a, n); return real_munmap(a,n); }
void * malloc(size_t n) { resolve(); void * p = real_malloc(n); if (n >= 16*1024*1024) trace("malloc",p,n); return p; }
void free(void * p) { resolve(); real_free(p); }
