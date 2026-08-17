# Printer setup: Brother QL-800

Good news — this ended up much simpler than earlier attempts. The QL-800
IS correctly installed as a normal Windows printer on this PC and prints
through it successfully (confirmed by an actual test print). So the app
now just sends the generated label through the normal Windows print
system, the same way any other app would print to it.

**No Zadig, no driver swap, no raw USB setup needed.** If you followed the
earlier Zadig instructions for this printer, you can ignore/revert that —
it isn't used anymore.

## What you need to check

1. **Printer name.** The app defaults to looking for a printer named
   exactly `Brother QL-800`. If yours shows up under a different exact
   name in Windows ("Devices and Printers"), set it explicitly:

   ```
   POST http://<pc-ip>:8000/printer
   Form field: printer_name = <exact name>
   ```

   Check what's currently configured / what Windows has installed via:

   ```
   GET http://<pc-ip>:8000/printer
   ```

2. **Label stock.** The label image is sized to whatever the printer
   driver reports as its printable area (via Windows' own DeviceCaps),
   which should automatically match whatever label size is configured in
   the printer's own Windows print preferences — the same settings used
   when you did your manual test print. Nothing to configure here as long
   as that's already set correctly.

## After pulling this update

Rebuild the launcher / let it re-provision (dependencies changed:
`brother_ql`/`pyusb`/`libusb-package` are gone, back to just `pywin32` +
`python-barcode`), then try Print from the tablet.
