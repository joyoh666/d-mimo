import argparse
import json
import logging
import os
from pathlib import Path
import platform
import sys
import time
import traceback

service = None
root_path = Path(__file__).absolute().parent

# Please setup path of tlkcore libraries to environment variables,
# here is a example to search from 'lib/' or '.'
prefix = "lib/"
lib_path = os.path.join(root_path, prefix)
if os.path.exists(lib_path):
    sys.path.insert(0, os.path.abspath(lib_path))
elif os.path.exists("tlkcore") and os.path.isdir("tlkcore"):
    print("Importing from source code")
else:
    print("Importing from Python site-packages")
    # If you want to import from Python site-packages, please remove the following line
    # sys.path.insert(0, os.path.abspath(root_path))

def check_ex_files(directory, extension=".so"):
    for file in os.listdir(directory):
        if file.endswith(extension):
            return True
    return False

try:
    from tlkcore import (
        TLKCoreService,
        DevInterface,
        RetCode,
        RFMode,
        UDState,
        UDMState,
        BeamType,
        UD_REF,
        UD_LO_CONFIG,
        RIS_Dir,
        RIS_ModuleConfig,
        CellRFMode,     # For CloverCell series AiP
        POLARIZATION,    # For CloverCell series AiP
        POLARIZATION_DEGSHIFT,  # For BBoxDuo series (legacy)
        POLARIZATION_TYPE,      # For BBoxDuo series (supports synthesis polarization)
        AzElAngle,
        ThetaPhiAngle
    )
except Exception as e:
    myos = platform.system()
    d = os.path.join(sys.path[0], 'tlkcore',)
    if ((myos == 'Windows' and check_ex_files(d, ".so"))
        or (myos == 'Linux' and check_ex_files(d, ".pyd"))):
        print(f"[Main] Import the wrong library for {myos}")
    else:
        print("[Main] Import path has something wrong")
        print(sys.path)
    traceback.print_exc()
    os._exit(-1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug.log')),
    ]
)

logger = logging.getLogger("Main")
logger.info("Python v%d.%d.%d (%s) on the %s platform" %(sys.version_info.major,
                                            sys.version_info.minor,
                                            sys.version_info.micro,
                                            platform.architecture()[0],
                                            platform.system()))

def startService(root:str=root_path, direct_connect_info:list=None, dfu_image:str=""):
    """
    Entry point: initialize TLKCoreService, connect device(s), then run testDevice().

    There are two connection modes:
      - Scan mode (default): automatically scans all available interfaces for devices.
      - Direct connect mode: skips scanning by accepting a known SN, IP, and device type.
        Use --dc SN Address DevType on the command line to activate this mode.

    USER CONFIGURATION — the variables below are the most common ones to change:
      - skip_builtin_tables : set True to skip loading built-in calibration and AAKit tables (not recommended for normal use)
      - interface           : choose which interface(s) to scan (see options below)
    """
    # ----------------------------------------------------------------
    # Resolve working root directory (for files/ and logs/ sub-folders)
    # ----------------------------------------------------------------
    if Path(root).exists() and Path(root).absolute() != Path(root_path):
        service_root = root
    else:
        service_root = root_path
    service = TLKCoreService(working_root = service_root)
    logger.info("TLKCoreService v%s %s" %(service.version, "is running" if service.running else "can not run"))

    if not service.running:
        return False

    # ----------------------------------------------------------------
    # USER CONFIGURATION — modify the values below to suit your needs
    # ----------------------------------------------------------------

    # Skip loading built-in tables (Calibration and AAKit) during initialization.
    #
    # Set to True to bypass the requirement for table files in the 'files/' directory.
    # Note:
    # - Beamforming functions will be UNAVAILABLE as they require calibration data.
    # - Single-channel control remains functional for manual gain/phase adjustments.
    # - Commonly used for initial hardware verification or when providing custom tables post-init.
    skip_builtin_tables = False

    # ----------------------------------------------------------------
    # Connection: Direct connect mode vs. Scan mode
    # ----------------------------------------------------------------

    if isinstance(direct_connect_info, list) and len(direct_connect_info) == 3:
        # --- Direct connect mode (activated via --dc SN Address DevType) ---
        # Use this when your device has a fixed/known address (e.g. static IP or COM port).
        # SN, address, and dev_type can be obtained from a previous scan result.
        kw = {
                'sn':                 direct_connect_info[0],  # Serial number string
                'address':            direct_connect_info[1],  # IP address or COM port (e.g. "192.168.100.2")
                'dev_type':           int(direct_connect_info[2]),  # Integer device type code
                'is_custom_calibration': skip_builtin_tables
            }
        ret = service.initDev(**kw)
        if ret.RetCode is RetCode.OK:
            kw['service'] = service
            kw['dfu_image'] = dfu_image
            kw.pop('is_custom_calibration', None) # Remove this parameter which is only for direct init
            testDevice(**kw)
    else:
        # --- Scan mode ---
        # Choose which interface(s) to scan. Combine with | or use DevInterface.ALL.
        # Options: DevInterface.LAN | DevInterface.COMPORT | DevInterface.ALL
        interface = DevInterface.ALL
        logger.info("Searching devices via: %s" %interface)
        ret = service.scanDevices(interface=interface)

        scan_dict = service.getScanInfo().RetData
        for i, (sn, (addr, devtype, in_dfu)) in enumerate(scan_dict.items(), start=1):
            logger.info("====== Dev_%d: %s, %s, %d, %r ======" %(i, sn, addr, devtype, in_dfu))

            # initDev() must be called before any other operation.
            # If initialization fails and the device is not already in DFU mode, skip it.
            if service.initDev(sn, is_custom_calibration=skip_builtin_tables).RetCode is not RetCode.OK and not in_dfu:
                continue
            testDevice(sn, service, dfu_image, addr, devtype, in_dfu)

    return True

def testDevice(sn, service, dfu_image:str="", address:str="", dev_type:int=0, in_dfu=False):
    """
    Query basic device info and dispatch to the appropriate test function.

    The test function is selected automatically based on the device type name returned by
    getDevTypeName(). The mapping is:  "test" + dev_name  → e.g. testBBox, testRIS, testUDM ...

    Supported device name suffixes (must match a function defined below):
        PD, UDBox, UDM, UDB, BBox, BBoard, CloverCell, RIS, BBoxDuo

    If dfu_image is provided, startDFU() is called instead of a test function.
    """
    dev_name = service.getDevTypeName(sn)

    fw_ver = None
    hw_ver = None
    loader_ver = None
    if in_dfu:
        logger.info("Device in DFU mode, you can skip error log for previous connection failed")
    else:
        logger.info("SN: %s" %service.querySN(sn))
        fw_ver = service.queryFWVer(sn).RetData
        hw_ver = service.queryHWVer(sn).RetData
        logger.info("FW ver: %s" %fw_ver)
        logger.info("HW ver: %s" %hw_ver)

    # Build the common keyword arguments passed to every test function
    kw = {}
    kw['sn'] = sn
    kw['service'] = service

    if len(dfu_image) > 0:
        # --- DFU path ---
        # Try to read bootloader version; if the device is already in DFU mode the query
        # will fail, which is expected — we just mark in_dfu=True and continue.
        if not in_dfu:
            ret = service.queryLoaderVer(sn)
            if ret.RetCode is not RetCode.OK:
                logger.warning("Error to query bootloader version: \'%s\', maybe it's in DFU mode" %ret.RetMsg)
                in_dfu = True
            else:
                loader_ver = ret.RetData
                logger.info(f"[DFU] Bootloader version: {loader_ver}")

        kw['dfu_image'] = dfu_image
        kw['dfu_dev_info'] = {
                            "sn": sn,
                            "address": address,
                            "dev_type": dev_type,
                            "in_dfu": in_dfu,
                            "fw_ver": fw_ver,
                            "hw_ver": hw_ver,
                            "loader_ver": loader_ver
                        }
        func_name = "startDFU"
    else:
        # --- Normal test path ---
        # Normalize device name to match one of the test* functions defined below.
        if 'BBoard' in dev_name:
            dev_name = "BBoard"
        elif 'BBox' in dev_name and 'Duo' not in dev_name:
            dev_name = "BBox"

        func_name = "test" + dev_name
        if func_name not in globals():
            logger.error("No test function found for device '%s' (looked for '%s'). "
                         "Please add a %s() function or update the name mapping above."
                         % (dev_name, func_name, func_name))
            service.DeInitDev(sn)
            return

    # Run the selected function, then release the device
    f = globals()[func_name]
    f(**kw)
    service.DeInitDev(sn)

""" ----------------- Test examples for TMY devices ----------------- """

__caliConfig = {
    "0.1GHz": {
            "lowPower": -35,
            "lowVolt": 34.68,
            "highPower": -5,
            "highVolt": 901.68
        },
    "0.3GHz": {
            "lowPower": -36,
            "lowVolt": 34.68,
            "highPower": -5,
            "highVolt": 901.68
        },
    "0.5GHz": {
            "lowPower": -36,
            "lowVolt": 109.98,
            "highPower": -5,
            "highVolt": 984.18
        },
    "1GHz": {
            "lowPower": -36,
            "lowVolt": 109.98,
            "highPower": -5,
            "highVolt": 984.18
        },
    "10GHz": {
            "lowPower": -36,
            "lowVolt": 57.6,
            "highPower": -5,
            "highVolt": 950.4
        },
    "20GHz": {
            "lowPower": -36,
            "lowVolt": 40.46,
            "highPower": -5,
            "highVolt": 936.36
        },
    "30GHz": {
            "lowPower": -36,
            "lowVolt": 83.81,
            "highPower": -5,
            "highVolt": 979.71
        },
    "40GHz": {
            "lowPower": -30,
            "lowVolt": 20.65,
            "highPower": -5,
            "highVolt": 787.65
        },
    "43GHz": {
            "lowPower": -28,
            "lowVolt": 20.65,
            "highPower": -5,
            "highVolt": 787.65
        }
}