--------------------------- MODULE CertRotation ---------------------------
(***************************************************************************)
(* Protocol P3 (docs/protocols.md): workload certificate renewal at T/2    *)
(* with issuance latency and failures/retries, discrete real time `now`,  *)
(* per-node clock offsets bounded by S (relative to real time), and CA    *)
(* rotation with trust-bundle overlap propagated to every verifier.       *)
(*                                                                         *)
(* One workload "w" presents its newest cert to every verifier.  Urgent    *)
(* actions (deliveries, due renewals, bundle-propagation deadlines) must   *)
(* fire before time advances.                                             *)
(***************************************************************************)
EXTENDS Integers, FiniteSets

CONSTANTS Verifiers,        \* peers that verify the workload's cert
          T,                \* cert lifetime
          S,                \* max |clock offset| of any node w.r.t. real time
          Lmax,             \* max issuance latency (ticks)
          R,                \* retry back-off after a failed issuance
          MaxFail,          \* max number of failed issuance attempts
          Horizon,          \* last real-time tick explored
          RenewDiv,         \* renew when local time >= notAfter - T \div RenewDiv (2 = at T/2)
          Backdate,         \* notBefore = CA local time - Backdate
          WaitBeforeRemove, \* old CA removed only once CA-local time >= max notAfter(old) + S
          WaitBundleBeforeSign \* new CA signs only after every verifier has it in its bundle

P == 2   \* bundle-update propagation deadline (ticks)

Nodes == {"ca", "w"} \cup Verifiers
CAs == {"ca1", "ca2"}

VARIABLES now, off, held, pend, fails, retryAt, issuer, bundle, rot, rotAt, ca1MaxNa

vars == <<now, off, held, pend, fails, retryAt, issuer, bundle, rot, rotAt, ca1MaxNa>>

Local(n) == now + off[n]
Newest == CHOOSE c \in held : \A d \in held : d.nb <= c.nb
Accepts(v, c) == c.ca \in bundle[v] /\ c.nb <= Local(v) /\ Local(v) <= c.na

\* Initial cert: issued by ca1 two ticks before time 0.
Init ==
  /\ now = 0
  /\ off \in [Nodes -> -S..S]
  /\ held = {[ca |-> "ca1", nb |-> -2 + off["ca"] - Backdate, na |-> -2 + off["ca"] + T]}
  /\ pend = [active |-> FALSE, at |-> 0, fail |-> FALSE]
  /\ fails = 0 /\ retryAt = 0
  /\ issuer = "ca1"
  /\ bundle = [v \in Verifiers |-> {"ca1"}]
  /\ rot = "pre" /\ rotAt = 0
  /\ ca1MaxNa = -2 + off["ca"] + T

RenewDue == ~pend.active /\ now >= retryAt /\ Local("w") >= Newest.na - (T \div RenewDiv)

Request ==
  /\ RenewDue /\ now < Horizon
  /\ \E l \in 1..Lmax, f \in (IF fails < MaxFail THEN BOOLEAN ELSE {FALSE}) :
       /\ pend' = [active |-> TRUE, at |-> now + l, fail |-> f]
       /\ fails' = fails + (IF f THEN 1 ELSE 0)
  /\ UNCHANGED <<now, off, held, retryAt, issuer, bundle, rot, rotAt, ca1MaxNa>>

Deliver ==
  /\ pend.active /\ pend.at = now
  /\ pend' = [pend EXCEPT !.active = FALSE]
  /\ IF pend.fail
     THEN /\ retryAt' = now + R
          /\ UNCHANGED <<held, ca1MaxNa>>
     ELSE /\ LET c == [ca |-> issuer, nb |-> Local("ca") - Backdate, na |-> Local("ca") + T]
             IN held' = held \cup {c}
          /\ ca1MaxNa' = IF issuer = "ca1" THEN Local("ca") + T ELSE ca1MaxNa
          /\ UNCHANGED retryAt
  /\ UNCHANGED <<now, off, fails, issuer, bundle, rot, rotAt>>

\* ---- CA rotation (environment) ----
Publish ==
  /\ rot = "pre" /\ now < Horizon
  /\ rot' = "published" /\ rotAt' = now
  /\ UNCHANGED <<now, off, held, pend, fails, retryAt, issuer, bundle, ca1MaxNa>>

AddToBundle(v) ==
  /\ rot \in {"published", "switched"} /\ "ca2" \notin bundle[v]
  /\ bundle' = [bundle EXCEPT ![v] = @ \cup {"ca2"}]
  /\ UNCHANGED <<now, off, held, pend, fails, retryAt, issuer, rot, rotAt, ca1MaxNa>>

Switch ==
  /\ rot = "published"
  /\ WaitBundleBeforeSign => \A v \in Verifiers : "ca2" \in bundle[v]
  /\ issuer' = "ca2" /\ rot' = "switched"
  /\ UNCHANGED <<now, off, held, pend, fails, retryAt, bundle, rotAt, ca1MaxNa>>

StartRemove ==
  /\ rot = "switched"
  /\ WaitBeforeRemove => Local("ca") >= ca1MaxNa + S
  /\ rot' = "removing" /\ rotAt' = now
  /\ UNCHANGED <<now, off, held, pend, fails, retryAt, issuer, bundle, ca1MaxNa>>

RemoveFromBundle(v) ==
  /\ rot = "removing" /\ "ca1" \in bundle[v]
  /\ bundle' = [bundle EXCEPT ![v] = @ \ {"ca1"}]
  /\ UNCHANGED <<now, off, held, pend, fails, retryAt, issuer, rot, rotAt, ca1MaxNa>>

\* Time advances only when nothing urgent is pending.
Urgent ==
  \/ (pend.active /\ pend.at = now)
  \/ RenewDue
  \/ (rot \in {"published", "switched"} /\ now >= rotAt + P /\ \E v \in Verifiers : "ca2" \notin bundle[v])
  \/ (rot = "removing" /\ now >= rotAt + P /\ \E v \in Verifiers : "ca1" \in bundle[v])

Tick ==
  /\ now < Horizon /\ ~Urgent
  /\ now' = now + 1
  /\ UNCHANGED <<off, held, pend, fails, retryAt, issuer, bundle, rot, rotAt, ca1MaxNa>>

Next ==
  \/ Request \/ Deliver \/ Tick \/ Publish \/ Switch \/ StartRemove
  \/ \E v \in Verifiers : AddToBundle(v) \/ RemoveFromBundle(v)

Spec == Init /\ [][Next]_vars

----------------------------------------------------------------------------
TypeOK ==
  /\ now \in 0..Horizon
  /\ held # {} /\ issuer \in CAs
  /\ \A v \in Verifiers : bundle[v] \subseteq CAs

\* R1: the workload always holds (and presents) a cert that every peer accepts.
R1 == \A v \in Verifiers : Accepts(v, Newest)

\* Sanity (expected to be VIOLATED): rotation completes -- old CA gone from every bundle and the
\* workload presents a ca2 cert -- proves rotation is reachable within the horizon (non-vacuity).
NotRotatedFully == ~(\A v \in Verifiers : bundle[v] = {"ca2"}) \/ Newest.ca # "ca2"

\* Weaker R1': some held cert is accepted by all peers (if the workload may pick which to present).
R1Weak == \E c \in held : \A v \in Verifiers : Accepts(v, c)
=============================================================================
