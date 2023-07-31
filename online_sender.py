#%%

from processing_chain.OnlineEncoding import OnlineEncoding
from communication.communicate_sender import sender

verbose = "cv2"
source =  "GoProWireless"

online_encoding = OnlineEncoding(source=source, verbose=verbose)

VERBOSE: bool = False
url: str = "tcp://localhost:5555"  # number specifies communication channel

sender(url=url, sender_processing=online_encoding, max_processing=-1, VERBOSE=VERBOSE)

del online_encoding


# %%
