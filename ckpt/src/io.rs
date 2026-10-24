//! Write back-ends for segment appends: io_uring (Linux) or a thread pool issuing `pwrite`.
//!
//! Both back-ends receive one contiguous buffer per group commit, split it into fixed-size pieces,
//! write the pieces concurrently at their offsets, then `fdatasync` the file.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;

pub const PIECE: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IoKind {
    Uring,
    Pwrite,
}

impl IoKind {
    pub fn parse(s: &str) -> Option<IoKind> {
        match s {
            "uring" => Some(IoKind::Uring),
            "pwrite" => Some(IoKind::Pwrite),
            _ => None,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            IoKind::Uring => "uring",
            IoKind::Pwrite => "pwrite",
        }
    }
}

pub trait Backend: Send {
    /// Write `buf` at `offset` (complete or error), then make it durable (fdatasync).
    fn write_durable(&mut self, f: &File, offset: u64, buf: &[u8]) -> io::Result<()>;
    fn kind(&self) -> IoKind;
}

pub struct Pwrite {
    threads: usize,
}

impl Pwrite {
    pub fn new(threads: usize) -> Self {
        Pwrite {
            threads: threads.max(1),
        }
    }
}

impl Backend for Pwrite {
    fn write_durable(&mut self, f: &File, offset: u64, buf: &[u8]) -> io::Result<()> {
        if buf.len() <= PIECE || self.threads == 1 {
            f.write_all_at(buf, offset)?;
        } else {
            let pieces: Vec<(usize, &[u8])> = buf
                .chunks(PIECE)
                .enumerate()
                .map(|(i, c)| (i * PIECE, c))
                .collect();
            let per = pieces.len().div_ceil(self.threads);
            std::thread::scope(|s| -> io::Result<()> {
                let hs: Vec<_> = pieces
                    .chunks(per)
                    .map(|grp| {
                        s.spawn(move || -> io::Result<()> {
                            for (o, c) in grp {
                                f.write_all_at(c, offset + *o as u64)?;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                for h in hs {
                    h.join()
                        .map_err(|_| io::Error::other("pwrite thread panicked"))??;
                }
                Ok(())
            })?;
        }
        f.sync_data()
    }
    fn kind(&self) -> IoKind {
        IoKind::Pwrite
    }
}

#[cfg(target_os = "linux")]
pub struct Uring {
    ring: io_uring::IoUring,
    depth: usize,
}

#[cfg(target_os = "linux")]
impl Uring {
    pub fn new(depth: u32) -> io::Result<Self> {
        Ok(Uring {
            ring: io_uring::IoUring::new(depth)?,
            depth: depth as usize,
        })
    }
}

#[cfg(target_os = "linux")]
impl Backend for Uring {
    fn write_durable(&mut self, f: &File, offset: u64, buf: &[u8]) -> io::Result<()> {
        use io_uring::{opcode, types};
        use std::os::unix::io::AsRawFd;
        let fd = types::Fd(f.as_raw_fd());
        let pieces: Vec<(usize, &[u8])> = buf
            .chunks(PIECE)
            .enumerate()
            .map(|(i, c)| (i * PIECE, c))
            .collect();
        let mut next = 0usize;
        let mut inflight = 0usize;
        let mut short: Vec<usize> = Vec::new();
        while next < pieces.len() || inflight > 0 {
            while next < pieces.len() && inflight < self.depth {
                let (o, c) = pieces[next];
                let e = opcode::Write::new(fd, c.as_ptr(), c.len() as u32)
                    .offset(offset + o as u64)
                    .build()
                    .user_data(next as u64);
                // SAFETY: buffers outlive the submission; we wait for every completion below.
                unsafe {
                    self.ring
                        .submission()
                        .push(&e)
                        .map_err(|_| io::Error::other("sq full"))?;
                }
                next += 1;
                inflight += 1;
            }
            self.ring.submit_and_wait(1)?;
            let done: Vec<(u64, i32)> = self
                .ring
                .completion()
                .map(|c| (c.user_data(), c.result()))
                .collect();
            for (ud, res) in done {
                inflight -= 1;
                let idx = ud as usize;
                if res < 0 {
                    return Err(io::Error::from_raw_os_error(-res));
                }
                if (res as usize) < pieces[idx].1.len() {
                    short.push(idx);
                }
            }
        }
        // Short writes are rare (e.g. signals); finish them synchronously.
        for idx in short {
            let (o, c) = pieces[idx];
            f.write_all_at(c, offset + o as u64)?;
        }
        let e = opcode::Fsync::new(fd)
            .flags(types::FsyncFlags::DATASYNC)
            .build()
            .user_data(u64::MAX);
        unsafe {
            self.ring
                .submission()
                .push(&e)
                .map_err(|_| io::Error::other("sq full"))?;
        }
        self.ring.submit_and_wait(1)?;
        let res = self
            .ring
            .completion()
            .next()
            .map(|c| c.result())
            .unwrap_or(-libc::EIO);
        if res < 0 {
            return Err(io::Error::from_raw_os_error(-res));
        }
        Ok(())
    }
    fn kind(&self) -> IoKind {
        IoKind::Uring
    }
}

/// Construct the requested back-end. `uring` falls back to `pwrite` (with a warning) where io_uring is
/// unavailable (non-Linux, or `io_uring_setup` refused); the effective kind is reported in STATS.
pub fn make(kind: IoKind, threads: usize) -> Box<dyn Backend> {
    match kind {
        IoKind::Pwrite => Box::new(Pwrite::new(threads)),
        IoKind::Uring => {
            #[cfg(target_os = "linux")]
            {
                match Uring::new(64) {
                    Ok(u) => return Box::new(u),
                    Err(e) => eprintln!("io_uring unavailable ({e}); falling back to pwrite"),
                }
            }
            #[cfg(not(target_os = "linux"))]
            eprintln!("io_uring requires Linux; falling back to pwrite");
            Box::new(Pwrite::new(threads))
        }
    }
}
