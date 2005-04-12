#!/usr/bin/python

if __name__ == "__main__":
    import os
    try:
        import wxPython
        import scanner.WXScannerLogCollector
        collector = WXScannerLogCollector.WXScannerLogCollector()
    except ImportError:
        import scanner.CLIScannerLogCollector
        collector = CLIScannerLogCollector.CLIScannerLogCollector()

    collector.run()

    # launch VNMR
    #os.spawnv( os.P_DETACH, "/vnmr/bin/vn" )

