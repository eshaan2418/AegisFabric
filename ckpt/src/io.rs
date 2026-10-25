//! I/O backends for segment appends and chunk reads.
//!
//! * `uring` (Linux only): one io_uring per backend; an append is a WRITE linked (IOSQE_IO_LINK) to an
//!   FDATASYNC in a single submission; reads are batched up to the ring depth.
//! * `pool` (portable fallback, selectable everywhere): pwrite + fdatasync on the calling thread; reads
//!   are spread over a scoped thread pool doing pread.
//!
//! Both are selected with `--io uring|pool`; the choice is reported in STATS.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

pub struct ReadReq {
    pub file: Arc<File>,
    pub off: u64,
    pub buf: Vec<u8>, // len = bytes to read
}

pub trait IoBackend: Send + Sync {
    fn name(&self) -> &'static str;
    /// Write `buf` at `off`, then make it durable (fdatasync) before returning.
    fn write_sync(&self, file: &File, off: u64, buf: &[u8]) -> io::Result<()>;
    /// Fill every request's buffer completely (short read = UnexpectedEof error for that call).
    fn read_many(&self, reqs: &mut [ReadReq]) -> io::Result<()>;
}

pub fn fdatasync(f: &File) -> io::Result<()> {
    f.sync_data()
}

pub struct PoolIo {
    threads: usize,
}

impl PoolIo {
    pub fn new(threads: usize) -> Self {
        PoolIo {
            threads: threads.max(1),
        }
    }
}

impl IoBackend for PoolIo {
    fn name(&self) -> &'static str {
        "pool"
    }
    fn write_sync(&self, file: &File, off: u64, buf: &[u8]) -> io::Result<()> {
        file.write_all_at(buf, off)?;
        fdatasync(file)
    }
    fn read_many(&self, reqs: &mut [ReadReq]) -> io::Result<()> {
        if reqs.len() <= 1 || self.threads == 1 {
            for r in reqs.iter_mut() {
                r.file.read_exact_at(&mut r.buf, r.off)?;
            }
            return Ok(());
        }
        let per = reqs.len().div_ceil(self.threads);
        std::thread::scope(|s| {
            let hs: Vec<_> = reqs
                .chunks_mut(per)
                .map(|part| {
                    s.spawn(move || -> io::Result<()> {
                        for r in part.iter_mut() {
                            r.file.read_exact_at(&mut r.buf, r.off)?;
                        }
                        Ok(())
                    })
                })
                .collect();
            for h in hs {
                h.join()
                    .map_err(|_| io::Error::other("reader thread panicked"))??;
            }
            Ok(())
        })
    }
}

#[cfg(target_os = "linux")]
pub use uring::UringIo;

#[cfg(target_os = "linux")]
mod uring {
    use super::*;
    use io_uring::{opcode, squeue, types, IoUring};
    use std::os::unix::io::AsRawFd;
    use std::sync::Mutex;

    pub struct UringIo {
        ring: Mutex<IoUring>,
        depth: u32,
    }

    impl UringIo {
        pub fn new(depth: u32) -> io::Result<Self> {
            Ok(UringIo {
                ring: Mutex::new(IoUring::new(depth)?),
                depth,
            })
        }
    }

    fn cqe_res(res: i32) -> io::Result<usize> {
        if res < 0 {
            Err(io::Error::from_raw_os_error(-res))
        } else {
            Ok(res as usize)
        }
    }

    impl IoBackend for UringIo {
        fn name(&self) -> &'static str {
            "uring"
        }

        fn write_sync(&self, file: &File, off: u64, buf: &[u8]) -> io::Result<()> {
            let fd = types::Fd(file.as_raw_fd());
            let mut ring = self.ring.lock().unwrap();
            let mut done = 0usize;
            loop {
                let rest = &buf[done..];
                // cap one SQE at 1 GiB (kernel limits single rw to ~2 GiB)
                let n = rest.len().min(1 << 30);
                let w = opcode::Write::new(fd, rest.as_ptr(), n as u32)
                    .offset(off + done as u64)
                    .build()
                    .flags(squeue::Flags::IO_LINK)
                    .user_data(1);
                let f = opcode::Fsync::new(fd)
                    .flags(types::FsyncFlags::DATASYNC)
                    .build()
                    .user_data(2);
                unsafe {
                    let mut sq = ring.submission();
                    sq.push(&w).map_err(|_| io::Error::other("sq full"))?;
                    sq.push(&f).map_err(|_| io::Error::other("sq full"))?;
                }
                ring.submit_and_wait(2)?;
                let mut wrote = 0usize;
                let mut sync_res = 0i32;
                for c in ring.completion() {
                    match c.user_data() {
                        1 => wrote = cqe_res(c.result())?,
                        _ => sync_res = c.result(),
                    }
                }
                done += wrote;
                if done >= buf.len() {
                    if sync_res == -libc::ECANCELED {
                        // fsync was cancelled because the linked write was short; sync now
                        return fdatasync(file);
                    }
                    cqe_res(sync_res)?;
                    return Ok(());
                }
                if wrote == 0 {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "io_uring write returned 0",
                    ));
                }
                // short write: linked fsync was cancelled (ECANCELED); loop and resubmit the rest
            }
        }

        fn read_many(&self, reqs: &mut [ReadReq]) -> io::Result<()> {
            let mut ring = self.ring.lock().unwrap();
            let mut filled = vec![0usize; reqs.len()];
            let mut pending: Vec<usize> = (0..reqs.len()).rev().collect();
            let mut inflight = 0usize;
            while !pending.is_empty() || inflight > 0 {
                while inflight < self.depth as usize {
                    let Some(i) = pending.pop() else { break };
                    let r = &mut reqs[i];
                    let rest = &mut r.buf[filled[i]..];
                    let e = opcode::Read::new(
                        types::Fd(r.file.as_raw_fd()),
                        rest.as_mut_ptr(),
                        rest.len() as u32,
                    )
                    .offset(r.off + filled[i] as u64)
                    .build()
                    .user_data(i as u64);
                    unsafe {
                        if ring.submission().push(&e).is_err() {
                            pending.push(i);
                            break;
                        }
                    }
                    inflight += 1;
                }
                ring.submit_and_wait(1)?;
                let done: Vec<(usize, i32)> = ring
                    .completion()
                    .map(|c| (c.user_data() as usize, c.result()))
                    .collect();
                for (i, res) in done {
                    inflight -= 1;
                    let n = cqe_res(res)?;
                    if n == 0 {
                        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "short read"));
                    }
                    filled[i] += n;
                    if filled[i] < reqs[i].buf.len() {
                        pending.push(i);
                    }
                }
            }
            Ok(())
        }
    }
}

/// Build a backend by name. `uring` is only available on Linux.
pub fn make(name: &str, threads: usize) -> anyhow::Result<Box<dyn IoBackend>> {
    match name {
        "pool" => Ok(Box::new(PoolIo::new(threads))),
        #[cfg(target_os = "linux")]
        "uring" => Ok(Box::new(UringIo::new(256)?)),
        #[cfg(not(target_os = "linux"))]
        "uring" => anyhow::bail!("io_uring is Linux-only; use --io pool"),
        "auto" => {
            #[cfg(target_os = "linux")]
            {
                if let Ok(u) = UringIo::new(256) {
                    return Ok(Box::new(u));
                }
            }
            Ok(Box::new(PoolIo::new(threads)))
        }
        other => anyhow::bail!("unknown io backend {other}"),
    }
}
