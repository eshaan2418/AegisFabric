---------------------------- MODULE CkptStore ----------------------------
(***************************************************************************)
(* Protocol P1 (docs/protocols.md): checkpoint chunk upload, group commit, *)
(* manifest COMMIT, WAL append/fsync, crash with loss of non-fsynced state *)
(* and torn WAL tail, recovery, and mark/sweep GC concurrent with commits. *)
(*                                                                         *)
(* Chunks are modelled as a set (one logical copy per chunk id).  RocksDB  *)
(* index writes are modelled as synchronous (durable on return).           *)
(* Flags select the deliberately-buggy variants and the proposed fixes.    *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Chunks, Mans, Refs, K, MaxCrash, MaxGC,
          BugIndexBeforeFsync,   \* index entry inserted at append time (before fsync)
          BugGcIgnoreEpoch,      \* sweep ignores the "indexed in epoch >= GC start" rule
          FixTouch,              \* COMMIT refreshes epoch of every referenced chunk
          FixLivePending         \* GC mark also treats accepted-but-unsynced WAL records as live

VARIABLES segVol,    \* chunks appended to the active segment, not yet fsync'd
          segDur,    \* chunks durably stored in segments
          pendIdx,   \* fsync'd chunks waiting for their index insert
          index,     \* chunks present in the (durable) RocksDB index
          idxEpoch,  \* epoch stamped on each index entry
          walDur,    \* durable WAL: sequence of [m, ok]; ok = FALSE is a torn record
          walVol,    \* WAL records appended but not fsync'd
          client,    \* per-manifest client state: "up" | "wait" | "done"
          acked,     \* chunks acknowledged to the client
          req,       \* chunks the client has sent and awaits ack for
          epoch, gc, crashes, gcs, up, recovered

vars == <<segVol, segDur, pendIdx, index, idxEpoch, walDur, walVol, client, acked, req,
          epoch, gc, crashes, gcs, up, recovered>>

NoGC == [active |-> FALSE, start |-> 0, live |-> {}, victims |-> {}]

\* Longest prefix of WAL records with valid checksums.
IntactLen(s) == IF \A i \in 1..Len(s) : s[i].ok THEN Len(s)
                ELSE (CHOOSE i \in 1..Len(s) : ~s[i].ok /\ \A j \in 1..(i-1) : s[j].ok) - 1
Intact(s) == SubSeq(s, 1, IntactLen(s))
MansIn(s) == { s[i].m : i \in 1..Len(s) }
LastK(s) == { s[i].m : i \in (IF Len(s) > K THEN Len(s) - K + 1 ELSE 1)..Len(s) }
RefsOf(S) == UNION { Refs[m] : m \in S }

Committed == MansIn(Intact(walDur))        \* checkpoint committed <=> durable intact WAL record
Retained  == LastK(Intact(walDur))         \* the K most recent committed manifests
PendingAccepted == { walVol[i] : i \in 1..Len(walVol) }

Init ==
  /\ segVol = {} /\ segDur = {} /\ pendIdx = {} /\ index = {}
  /\ idxEpoch = [c \in Chunks |-> 0]
  /\ walDur = <<>> /\ walVol = <<>>
  /\ client = [m \in Mans |-> "up"] /\ acked = [m \in Mans |-> {}] /\ req = [m \in Mans |-> {}]
  /\ epoch = 0 /\ gc = NoGC /\ crashes = 0 /\ gcs = 0 /\ up = TRUE
  /\ recovered = [m |-> "none", err |-> FALSE]

\* PUT_CHUNK: dedup hit is acked immediately; otherwise append to the active segment.
Put(m, c) ==
  /\ up /\ client[m] = "up" /\ c \in Refs[m] \ (acked[m] \cup req[m])
  /\ IF c \in index
     THEN /\ acked' = [acked EXCEPT ![m] = @ \cup {c}]
          /\ UNCHANGED <<segVol, index, idxEpoch, req>>
     ELSE /\ segVol' = segVol \cup {c}
          /\ req' = [req EXCEPT ![m] = @ \cup {c}]
          /\ IF BugIndexBeforeFsync
             THEN /\ index' = index \cup {c}
                  /\ idxEpoch' = [idxEpoch EXCEPT ![c] = epoch]
             ELSE UNCHANGED <<index, idxEpoch>>
          /\ UNCHANGED acked
  /\ UNCHANGED <<segDur, pendIdx, walDur, walVol, client, epoch, gc, crashes, gcs, up, recovered>>

\* Group commit: fsync the segment; buffered chunks become durable.
Fsync ==
  /\ up /\ segVol # {}
  /\ segDur' = segDur \cup segVol /\ pendIdx' = pendIdx \cup segVol /\ segVol' = {}
  /\ UNCHANGED <<index, idxEpoch, walDur, walVol, client, acked, req, epoch, gc, crashes, gcs, up, recovered>>

\* Index insert happens only after the fsync above.
IndexInsert(c) ==
  /\ up /\ c \in pendIdx
  /\ index' = index \cup {c} /\ idxEpoch' = [idxEpoch EXCEPT ![c] = epoch]
  /\ pendIdx' = pendIdx \ {c}
  /\ UNCHANGED <<segVol, segDur, walDur, walVol, client, acked, req, epoch, gc, crashes, gcs, up, recovered>>

\* Ack only once the chunk is indexed.
Ack(m, c) ==
  /\ up /\ client[m] = "up" /\ c \in req[m] \ acked[m] /\ c \in index
  /\ acked' = [acked EXCEPT ![m] = @ \cup {c}]
  /\ UNCHANGED <<segVol, segDur, pendIdx, index, idxEpoch, walDur, walVol, client, req, epoch, gc,
                 crashes, gcs, up, recovered>>

\* Client timeout: a sent chunk that the server no longer holds (e.g. swept by GC between index
\* insert and ack delivery) is re-sent.
Resend(m, c) ==
  /\ up /\ client[m] = "up" /\ c \in req[m] \ acked[m]
  /\ c \notin index \cup pendIdx \cup segVol
  /\ req' = [req EXCEPT ![m] = @ \ {c}]
  /\ UNCHANGED <<segVol, segDur, pendIdx, index, idxEpoch, walDur, walVol, client, acked, epoch, gc,
                 crashes, gcs, up, recovered>>

\* COMMIT: rejected unless every referenced chunk is indexed (client then re-uploads).
Commit(m) ==
  /\ up /\ client[m] = "up" /\ acked[m] = Refs[m]
  /\ IF Refs[m] \subseteq index
     THEN /\ walVol' = Append(walVol, m)
          /\ client' = [client EXCEPT ![m] = "wait"]
          /\ idxEpoch' = IF FixTouch
                         THEN [c \in Chunks |-> IF c \in Refs[m] THEN epoch ELSE idxEpoch[c]]
                         ELSE idxEpoch
          /\ UNCHANGED <<acked, req>>
     ELSE /\ acked' = [acked EXCEPT ![m] = {}] /\ req' = [req EXCEPT ![m] = {}]
          /\ UNCHANGED <<walVol, client, idxEpoch>>
  /\ UNCHANGED <<segVol, segDur, pendIdx, index, walDur, epoch, gc, crashes, gcs, up, recovered>>

WalFsync ==
  /\ up /\ walVol # <<>>
  /\ walDur' = walDur \o [i \in 1..Len(walVol) |-> [m |-> walVol[i], ok |-> TRUE]]
  /\ walVol' = <<>>
  /\ UNCHANGED <<segVol, segDur, pendIdx, index, idxEpoch, client, acked, req, epoch, gc, crashes,
                 gcs, up, recovered>>

\* Ack the commit to the client only once its WAL record is durable.
CommitAck(m) ==
  /\ up /\ client[m] = "wait" /\ m \in Committed
  /\ client' = [client EXCEPT ![m] = "done"]
  /\ UNCHANGED <<segVol, segDur, pendIdx, index, idxEpoch, walDur, walVol, acked, req, epoch, gc,
                 crashes, gcs, up, recovered>>

\* GC mark: snapshot the live set, start a new epoch.
GcMark ==
  /\ up /\ ~gc.active /\ gcs < MaxGC
  /\ LET live == RefsOf(Retained) \cup (IF FixLivePending THEN RefsOf(PendingAccepted) ELSE {})
     IN gc' = [active |-> TRUE, start |-> epoch + 1, live |-> live, victims |-> index \ live]
  /\ epoch' = epoch + 1 /\ gcs' = gcs + 1
  /\ UNCHANGED <<segVol, segDur, pendIdx, index, idxEpoch, walDur, walVol, client, acked, req,
                 crashes, up, recovered>>

\* GC sweep, one chunk at a time (atomic w.r.t. PUT/COMMIT on the same index entry).
\* Segment compaction (copy live chunks, fsync, delete victim segment) is abstracted as atomic
\* removal of the dead chunk.
GcSweep(c) ==
  /\ up /\ gc.active /\ c \in gc.victims
  /\ IF c \in index /\ (BugGcIgnoreEpoch \/ idxEpoch[c] < gc.start)
     \* Only the indexed (old) copy is reclaimed; a newer fsync'd copy awaiting its index insert
     \* (c \in pendIdx) lives in a different segment and survives.
     THEN /\ index' = index \ {c}
          /\ segDur' = IF c \in pendIdx THEN segDur ELSE segDur \ {c}
     ELSE UNCHANGED <<index, segDur>>
  /\ gc' = [gc EXCEPT !.victims = @ \ {c}]
  /\ UNCHANGED <<segVol, pendIdx, idxEpoch, walDur, walVol, client, acked, req, epoch, crashes, gcs,
                 up, recovered>>

GcDone ==
  /\ up /\ gc.active /\ gc.victims = {}
  /\ gc' = NoGC
  /\ UNCHANGED <<segVol, segDur, pendIdx, index, idxEpoch, walDur, walVol, client, acked, req, epoch,
                 crashes, gcs, up, recovered>>

\* Crash (kill -9 / power loss): any subset of un-fsync'd chunks and any prefix of un-fsync'd WAL
\* records may have reached the disk; the next WAL record may be torn.
Crash ==
  /\ up /\ crashes < MaxCrash
  /\ \E surv \in SUBSET segVol, n \in 0..Len(walVol), torn \in BOOLEAN :
       /\ segDur' = segDur \cup surv
       /\ walDur' = walDur \o [i \in 1..n |-> [m |-> walVol[i], ok |-> TRUE]]
                           \o (IF torn /\ n < Len(walVol) THEN <<[m |-> walVol[n + 1], ok |-> FALSE]>> ELSE <<>>)
  /\ segVol' = {} /\ walVol' = <<>> /\ pendIdx' = {} /\ gc' = NoGC
  /\ up' = FALSE /\ crashes' = crashes + 1
  /\ UNCHANGED <<index, idxEpoch, client, acked, req, epoch, gcs, recovered>>

\* Recovery: truncate torn tail, drop index entries whose data did not survive, report latest.
Recover ==
  /\ ~up
  /\ LET w == Intact(walDur) IN
       /\ walDur' = w
       /\ index' = index \cap segDur
       /\ recovered' = IF Len(w) = 0 THEN [m |-> "none", err |-> FALSE]
                       ELSE [m |-> w[Len(w)].m, err |-> ~(Refs[w[Len(w)].m] \subseteq segDur)]
       /\ client' = [m \in Mans |-> IF client[m] = "done" \/ (client[m] = "wait" /\ m \in MansIn(w))
                                    THEN "done" ELSE "up"]
  /\ acked' = [m \in Mans |-> {}] /\ req' = [m \in Mans |-> {}]
  /\ up' = TRUE
  /\ UNCHANGED <<segVol, segDur, pendIdx, idxEpoch, walVol, epoch, gc, crashes, gcs>>

Next ==
  \/ \E m \in Mans, c \in Chunks : Put(m, c) \/ Ack(m, c) \/ Resend(m, c)
  \/ \E m \in Mans : Commit(m) \/ CommitAck(m)
  \/ \E c \in Chunks : IndexInsert(c) \/ GcSweep(c)
  \/ Fsync \/ WalFsync \/ GcMark \/ GcDone \/ Crash \/ Recover

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

----------------------------------------------------------------------------
TypeOK ==
  /\ segVol \subseteq Chunks /\ segDur \subseteq Chunks /\ index \subseteq Chunks
  /\ client \in [Mans -> {"up", "wait", "done"}]

\* I1: no retained committed manifest references a chunk that is not durably stored (and, while
\* the server is up, indexed).
I1 == \A m \in Retained : Refs[m] \subseteq segDur /\ (up => Refs[m] \subseteq index)

\* I2: recovery returns the latest committed manifest (by construction: last intact WAL record) and
\* every chunk of it is intact, i.e. the "detected error" branch is never needed in a correct store.
I2 == recovered.m # "none" => (~recovered.err /\ recovered.m \in Committed)

\* No commit acknowledged to a client is ever lost.
AckedDurable == \A m \in Mans : client[m] = "done" => m \in Committed

\* I3: GC never removes a chunk of a retained committed manifest or of an accepted in-flight commit.
I3 == up => \A m \in Retained \cup PendingAccepted : Refs[m] \subseteq segDur

\* Sanity (expected to be VIOLATED): a crash and a full GC cycle happen and both checkpoints still
\* get committed -- proves the interesting interleavings are reachable (non-vacuity).
NotAllDoneAfterCrashAndGC ==
  ~(crashes = MaxCrash /\ gcs = MaxGC /\ ~gc.active /\ \A m \in Mans : client[m] = "done")

\* Progress: every checkpoint is eventually committed and acknowledged.
AllDone == <>(\A m \in Mans : client[m] = "done")
=============================================================================
