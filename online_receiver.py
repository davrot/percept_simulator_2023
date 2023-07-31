#%%
from communication.communicate_receiver import receiver
from processing_chain.OnlinePerception import OnlinePerception

online_perception = OnlinePerception("cv2", use_gui=True)

url: str = "tcp://*:5555"  # number specifies communication channel

receiver(url, online_perception, VERBOSE=False)

# del online_perception


# %%
