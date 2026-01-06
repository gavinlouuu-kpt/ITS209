#include "storage.h"
#include <LittleFS.h>
#include <Preferences.h>

// Configuration
static constexpr int MAX_FORMAT_ATTEMPTS = 3;
static const char *PREF_NAMESPACE = "fs";
static const char *PREF_FMT_ATTEMPTS = "fmt_attempts";

// Try to mount LittleFS. On failure, perform one controlled format attempt
// (incrementing a persisted counter in NVS) and retry mounting. The function
// ensures we don't loop formatting the flash by respecting MAX_FORMAT_ATTEMPTS.
bool initLittleFS()
{
    Preferences prefs;
    bool prefsOpened = prefs.begin(PREF_NAMESPACE, false);
    if (!prefsOpened)
    {
        Serial.println("WARN: Preferences begin() failed; proceeding without persisted counters");
    }

    int formatAttempts = 0;
    if (prefsOpened)
    {
        formatAttempts = prefs.getInt(PREF_FMT_ATTEMPTS, 0);
    }

    // First try to mount without formatting.
    if (LittleFS.begin())
    {
        if (prefsOpened)
        {
            prefs.putInt(PREF_FMT_ATTEMPTS, 0);
            prefs.end();
        }
        Serial.println("LittleFS: mounted successfully");
        return true;
    }

    Serial.printf("LittleFS: mount failed (attempts=%d)\n", formatAttempts);

    // Prevent endless format loops across reboots.
    if (formatAttempts >= MAX_FORMAT_ATTEMPTS)
    {
        Serial.println("LittleFS: reached max format attempts; aborting recovery");
        if (prefsOpened)
            prefs.end();
        return false;
    }

    // Persist that we are about to attempt a format (one controlled attempt).
    if (prefsOpened)
    {
        prefs.putInt(PREF_FMT_ATTEMPTS, formatAttempts + 1);
        prefs.end();
    }

    // Perform format and retry mount.
    Serial.println("LittleFS: attempting to format filesystem to recover...");
    bool formatted = LittleFS.format();
    if (!formatted)
    {
        Serial.println("LittleFS: format() returned false (may still succeed to mount)");
    }

    // Small pause to let the underlying driver settle.
    delay(100);

    if (LittleFS.begin())
    {
        // Clear persisted counter on success.
        Preferences prefs2;
        if (prefs2.begin(PREF_NAMESPACE, false))
        {
            prefs2.putInt(PREF_FMT_ATTEMPTS, 0);
            prefs2.end();
        }
        Serial.println("LittleFS: mounted successfully after format");
        return true;
    }

    Serial.println("LittleFS: mount still failing after format");
    return false;
}

