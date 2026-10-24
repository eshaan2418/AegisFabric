---- MODULE MC ----
EXTENDS CkptStore
\* Two manifests sharing chunk c2 (exercises dedup); m2 alone owns c3.
MCRefs == [m \in {"m1", "m2"} |-> IF m = "m1" THEN {"c1", "c2"} ELSE {"c2", "c3"}]
====
