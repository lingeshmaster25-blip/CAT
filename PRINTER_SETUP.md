# One-time setup: Brother QL-800 direct USB printing

Your QL-800 currently only prints through P-touch Editor Lite, which talks
to it using Brother's own protocol — NOT the normal Windows print system.
So instead of printing through Windows, the server now talks to the
printer directly over USB using `brother_ql`, an open-source library built
specifically for Brother QL-series printers. This bypasses P-touch Editor
entirely.

For this to work, Windows needs to let a Python library claim the
printer's USB connection directly, instead of whatever currently has it
(likely nothing formal — P-touch Editor Lite probably talks to it in an
ad-hoc way). This requires a **one-time driver swap** using a free tool
called Zadig. This only needs to be done once on the PC running the OCR
server.

## Steps

1. **Close P-touch Editor Lite** if it's open, and make sure the QL-800 is
   connected via USB and powered on.

2. Download Zadig: https://zadig.akeo.ie/ (no install needed, just run the
   .exe).

3. Open Zadig. In the top menu, go to **Options → List All Devices** (this
   makes sure the QL-800 shows up even if Windows doesn't see it as a
   standard device).

4. In the dropdown, find the QL-800. It might show up as "QL-800",
   "Brother QL-800", or just a generic printer/USB name — if you're not
   sure which entry it is, unplug the printer, see which entry disappears
   from the list, then plug it back in.

5. To the right of the dropdown, you'll see a driver arrow like:
   `(NULL) → WinUSB`
   Make sure **WinUSB** is selected as the target driver (it usually is by
   default).

6. Click **Replace Driver** (or **Install Driver** if that's what it says).
   Wait for it to finish — this can take a minute.

7. That's it. The QL-800 will no longer show up as a normal
   printer/P-touch device in the same way, but the OCR server can now talk
   to it directly.

## Verifying it worked

Once the server's been rebuilt with the new dependencies (see below), you
can check the connection without printing anything:

```
GET http://<pc-ip>:8000/printer
```

This returns what's configured and what brother_ql can currently discover
on USB. If `discovered` is empty, the driver swap likely didn't take, or
the printer isn't connected/powered on.

## If you ever need to print via P-touch Editor Lite again

The Zadig driver swap is specific to this one USB device slot. If you
plug the QL-800 into a different USB port later, Windows may treat it as
a "new" device and revert to its original behavior until you Zadig it
again on that port too. If you genuinely need P-touch Editor Lite back on
this exact port, Zadig also lets you revert via **Device Manager →
Uninstall device**, then unplug/replug the printer.
