#!/usr/bin/env python
"""Decode an X86EMU_TRACE_SYSCALLS trace into named syscalls.

The trace writes the `syscall` instruction itself and the instruction after
it, so each pair is one call: RAX/RDI/RSI/RDX/R10/R8/R9 going in, RAX coming
back.  Threads are told apart by their stack pointer, which is the only
thread identity a register dump carries.

Usage: python tools/systrace.py [trace.txt] [-n LAST]
"""
import sys, re

SYS = {0:'read',1:'write',2:'open',3:'close',4:'stat',5:'fstat',6:'lstat',
 7:'poll',8:'lseek',9:'mmap',10:'mprotect',11:'munmap',12:'brk',
 13:'rt_sigaction',14:'rt_sigprocmask',15:'rt_sigreturn',16:'ioctl',
 17:'pread64',18:'pwrite64',19:'readv',20:'writev',21:'access',22:'pipe',
 23:'select',24:'sched_yield',25:'mremap',28:'madvise',32:'dup',33:'dup2',
 35:'nanosleep',39:'getpid',40:'sendfile',41:'socket',42:'connect',
 43:'accept',44:'sendto',45:'recvfrom',46:'sendmsg',47:'recvmsg',
 48:'shutdown',49:'bind',50:'listen',51:'getsockname',54:'setsockopt',
 55:'getsockopt',56:'clone',57:'fork',58:'vfork',59:'execve',60:'exit',
 61:'wait4',62:'kill',63:'uname',72:'fcntl',73:'flock',74:'fsync',
 78:'getdents',79:'getcwd',80:'chdir',82:'rename',83:'mkdir',84:'rmdir',
 85:'creat',86:'link',87:'unlink',88:'symlink',89:'readlink',90:'chmod',
 91:'fchmod',92:'chown',95:'umask',96:'gettimeofday',97:'getrlimit',
 99:'sysinfo',100:'times',102:'getuid',104:'getgid',105:'setuid',
 107:'geteuid',108:'getegid',109:'setpgid',110:'getppid',112:'setsid',
 132:'utime',133:'mknod',137:'statfs',138:'fstatfs',158:'arch_prctl',
 160:'setrlimit',161:'chroot',165:'mount',166:'umount2',169:'reboot',
 186:'gettid',200:'tkill',201:'time',202:'futex',213:'epoll_create',
 217:'getdents64',218:'set_tid_address',228:'clock_gettime',
 229:'clock_getres',230:'clock_nanosleep',231:'exit_group',232:'epoll_wait',
 233:'epoll_ctl',234:'tgkill',257:'openat',258:'mkdirat',259:'mknodat',
 260:'fchownat',262:'newfstatat',263:'unlinkat',264:'renameat',
 265:'linkat',266:'symlinkat',267:'readlinkat',268:'fchmodat',
 269:'faccessat',270:'pselect6',271:'ppoll',272:'unshare',
 273:'set_robust_list',274:'get_robust_list',280:'utimensat',
 281:'epoll_pwait',282:'signalfd',283:'timerfd_create',284:'eventfd',
 285:'fallocate',286:'timerfd_settime',287:'timerfd_gettime',
 288:'accept4',290:'eventfd2',291:'epoll_create1',292:'dup3',293:'pipe2',
 294:'inotify_init1',295:'preadv',296:'pwritev',302:'prlimit64',
 316:'renameat2',318:'getrandom',319:'memfd_create',332:'statx',
 334:'rseq',435:'clone3',439:'faccessat2',441:'epoll_pwait2',
 452:'fchmodat2'}

FUTEX_OPS = {0:'WAIT',1:'WAKE',2:'FD',3:'REQUEUE',4:'CMP_REQUEUE',5:'WAKE_OP',
 6:'LOCK_PI',7:'UNLOCK_PI',8:'TRYLOCK_PI',9:'WAIT_BITSET',10:'WAKE_BITSET',
 11:'WAIT_REQUEUE_PI',12:'CMP_REQUEUE_PI'}

def futex_str(op):
    base = op & 0x7F
    s = FUTEX_OPS.get(base, str(base))
    if op & 128: s += '|PRIV'
    if op & 256: s += '|CLKRT'
    return s

R = re.compile(r'\[(\d+)\] rip=([0-9A-F]+) bytes=([0-9A-F]{2}) ([0-9A-F]{2}).*?'
               r'rax=([0-9A-F]+) rcx=[0-9A-F]+ rdx=([0-9A-F]+) rbx=[0-9A-F]+ '
               r'rsp=([0-9A-F]+) rbp=[0-9A-F]+ rsi=([0-9A-F]+) rdi=([0-9A-F]+) '
               r'r8=([0-9A-F]+) r9=([0-9A-F]+) r10=([0-9A-F]+)')

def main():
    path = 'trace.txt'
    last = 80
    args = sys.argv[1:]
    skip = False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a == '-n':
            last = int(args[i+1]); skip = True
        elif not a.startswith('-'):
            path = a
    calls = []
    pending = None
    for line in open(path):
        m = R.match(line)
        if not m: continue
        n, rip, b0, b1, rax, rdx, rsp, rsi, rdi, r8, r9, r10 = m.groups()
        if b0 == '0F' and b1 == '05':
            pending = (int(n), int(rax,16), int(rdi,16), int(rsi,16),
                       int(rdx,16), int(r10,16), int(r8,16), int(r9,16),
                       int(rsp,16), rip)
        elif pending:
            calls.append(pending + (int(rax,16),))
            pending = None
    if pending:
        calls.append(pending + (None,))   # never returned

    stacks = {}
    for c in calls[-last:]:
        (n, nr, a1, a2, a3, a4, a5, a6, rsp, rip, ret) = c
        tid = stacks.setdefault(rsp >> 20, 'T%d' % (len(stacks)+1))
        name = SYS.get(nr, 'sys_%d' % nr)
        if nr == 202:
            argstr = 'uaddr=%X op=%s val=%X timeout=%s' % (
                a1, futex_str(a2), a3 & 0xFFFFFFFF, 'NULL' if a4 == 0 else '%X' % a4)
        else:
            argstr = '%X, %X, %X' % (a1, a2, a3)
        r = 'BLOCKED (never returned)' if ret is None else (
            '-%d' % (0x10000000000000000 - ret) if ret > 0xFFFFFFFF00000000 else '%X' % ret)
        print('[%d] %-4s %-16s %-58s = %s' % (n, tid, name, argstr, r))

main()
