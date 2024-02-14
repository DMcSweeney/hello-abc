"""
Launch a DICOM storage SCP
"""

from pydicom.uid import ExplicitVRLittleEndian

from pynetdicom import AE, debug_logger, evt
from pynetdicom.sop_class import CTImageStorage



def handle_store(event):
    #* Handle EVT_C_STORE events
    ds = event.dataset
    ds.file_meta = event.file_meta
    assert 'SOPInstanceUID' in ds
    ds.save_as(ds.SOPInstanceUID, write_like_original=False)

    return 0x0000


def main():
    debug_logger()
    handlers = [(evt.EVT_C_STORE, handle_store)]

    ae = AE()
    ae.add_supported_context(CTImageStorage, ExplicitVRLittleEndian)
    print("Starting DICOM SCP", flush=True)
    ae.start_server(("127.0.0.1", 11112), block=True, evt_handlers=handlers)

if __name__ == '__main__':
    main()