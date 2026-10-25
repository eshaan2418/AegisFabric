//! AegisFabric checkpoint engine (protocol P1 in docs/protocols.md).
//!
//! * `format`  — segment / WAL / manifest encodings
//! * `chunker` — fixed-size and content-defined chunking
//! * `io`      — io_uring (Linux) and thread-pool pwrite backends
//! * `store`   — log-structured chunk store, RocksDB index, manifest WAL, recovery, GC
//! * `server`  — TCP protocol used by `ckpt/client.py`

pub mod chunker;
pub mod format;
pub mod io;
pub mod server;
pub mod store;
