#%%
import zmq
import time

from communication.communicate_datapacket import DataPacket


def receiver(url, receiver_processing, VERBOSE=False):

    with zmq.Context() as context:

        # establishing connection...
        socket = context.socket(zmq.REP)
        socket.bind(url)
        data_in = None
        data_out_cache = None

        while socket.closed is False:

            # MAIN JOB: here we do a fast task!
            if VERBOSE:
                print("RECEIVER: doing fast main job")
            data_out = receiver_processing.update(data_in)
            data_in = None  # clear after processing!
            if data_out is not None:
                data_out_cache = data_out

            # CHECK: is packet available?
            if VERBOSE:
                print("RECEIVER: Now looking for packets...")
            try:
                packet_in = socket.recv_pyobj(zmq.NOBLOCK)
                packet_available = True
                if VERBOSE:
                    print("   ...packet available!")
            except zmq.error.Again:
                packet_available = False
                if VERBOSE:
                    print("   ...nothing found")

            flag_stop = False
            if packet_available:

                # YES, packet is available - interpret it!
                if packet_in.message == "stop":
                    flag_stop = True
                    if VERBOSE:
                        print("RECEIVER: I have to stop this program!")
                if packet_in.message == "sender":
                    data_in = packet_in.data

                if VERBOSE:
                    print("RECEIVER: Confirm data packet was received and processed.")
                packet_out = DataPacket(data=data_out_cache, message="receiver")
                print(packet_out)
                socket.send_pyobj(packet_out)
                data_out_cache = None

            if flag_stop:
                if VERBOSE:
                    print("RECEIVER: Trying to close socket!")
                socket.close()

        if VERBOSE:
            print("RECEIVER: Socket is closed!")
    if VERBOSE:
        print("RECEIVER: Context seems to have been released!")


if __name__ == "__main__":

    from test_class_animate import TestClassAnimate

    # count = None
    url: str = "tcp://*:5555"  # number specifies communication channel

    class_processing = TestClassAnimate(show_vertical_bar=True, wait_interval=0.1)
    receiver(url, class_processing, VERBOSE=True)
    del class_processing


# %%
