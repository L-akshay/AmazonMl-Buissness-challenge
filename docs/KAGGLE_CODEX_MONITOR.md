# Automatic follow-up in the existing Codex conversation

`scripts/watch_kaggle_codex.py` reads the controller journal every 60 seconds
and uses the installed, supported `codex queue --thread ... --message ...`
command to queue an event in the user-selected existing conversation. It does
not create another agent, launch Kaggle jobs, expose a webhook, or train locally.

Events cover job status changes, read errors, downloaded deliverables, and a
controller journal that has stopped updating for five minutes. An unchanged
healthy run gets a follow-up every 15 minutes. Events are deduplicated across
watcher restarts; remote error text and dataset contents are excluded from
messages. An OS file lock prevents two watchers for this project. A queued
message is evidence of acceptance, not proof that Codex has already handled it.

The monitor runs for 24 hours by default, or stops after all planned jobs and
requested downloads finish. The computer must remain awake, with Codex available
and network access. This is a local process, not an always-on cloud service;
Windows restart or shutdown stops it. A watcher cannot revive itself after its
own process dies. The journal and monitor state survive restart.

Start from this repository in PowerShell, substituting an explicitly verified
existing conversation ID:

```powershell
.venv\Scripts\python.exe scripts/watch_kaggle_codex.py --codex (Get-Command codex).Source --thread '<conversation-id>' --hours 24
```

For unattended use, launch hidden with `Start-Process -WindowStyle Hidden`,
redirect stdout/stderr into `cache/managed_kaggle`, and retain its PID. Check
`codex_monitor_health.json` for the latest heartbeat, `codex_monitor_state.json`
for the last accepted event, and `codex-monitor.out.log` for event delivery.
Stopping this watcher does not stop the controller or remote jobs.

The controller remains the single job launcher. On an alert, Codex should verify
live remote status, preserve checkpoints, investigate the cause, and recover
within the existing user authorization. An ambiguous launch must be reconciled
before retrying; alerts do not authorize duplicate work or reduced quality.

Validation on 2026-09-27: the installed CLI accepted a connectivity test and the
watcher's initial event for the existing conversation. The watcher heartbeat
subsequently advanced without sending duplicate initial events. The bounded
local suite ran 52 tests (51 passed, one skipped), peaking at 0.216 GiB process
tree RSS under the existing two-CPU/3-GiB guard. Actual idle-thread response to
the queued event remains an end-to-end observation to confirm after this turn.
