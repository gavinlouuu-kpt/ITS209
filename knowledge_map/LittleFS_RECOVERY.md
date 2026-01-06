# LittleFS auto-recover behavior (implemented)

This document describes the automatic LittleFS recovery behavior added to the project.

Behavior summary
- On boot, `initLittleFS()` is called early in `setup()` to attempt to mount LittleFS.
- If mounting fails, the code will attempt a controlled format of LittleFS and retry mounting.
- A persisted counter (`fmt_attempts`) is stored in NVS (`Preferences`) under namespace `fs` to prevent repeated formats across reboots.
- Formatting attempts are limited to `MAX_FORMAT_ATTEMPTS` (3 by default).
- On a successful mount the counter is cleared. If the mount still fails after the limit, the device will abort storage initialization and remain in degraded mode.

Files added
- `include/storage.h` — public header declaring `initLittleFS()`
- `src/storage.cpp` — implementation using `LittleFS` and `Preferences`

How to test
1. Reproduce the Corrupted dir pair error (e.g. force LittleFS mount failure or corrupt fs on a test device).
2. Reboot device and observe serial logs: expect `mount failed -> attempting to format -> mounted successfully` or `reached max format attempts`.
3. Confirm `fmt_attempts` persists across reboots until a successful mount clears it.

Notes and safety
- This mode will erase LittleFS contents on format. Use only when data loss is acceptable or when you prefer automatic recovery.
- The format-attempt counter prevents wear from repeated formats across boot loops.
- If you prefer an alternative policy (repair-only, read-only fallback, or backup storage), update `initLittleFS()` and the startup logic accordingly.



