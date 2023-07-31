#%%
import zmq
import time
import math

from communication.communicate_datapacket import DataPacket
from communication.test_class_animate import TestClassAnimate


def sender(url, sender_processing, max_processing=math.inf, VERBOSE=False):

    context = zmq.Context()

    socket = context.socket(zmq.REQ)
    with socket.connect(url):

        try:
            cur_processing = 0
            running = True
            data_in = None
            while running:

                # MAIN JOB: the following function performs heavy
                # data processing and sends a result to the receiving process
                data_out = sender_processing.update(data_in)
                packet_out = DataPacket(data=data_out, message="sender")
                socket.send_pyobj(packet_out)

                # CONFIRMATION: Waits indefinitely until receiving a
                # confirmation data packet. Can be stopped by pressing
                # CTRL-C, which is handled below as exception
                if VERBOSE:
                    print("SENDER: Sent, waiting for receiver confirmation...")
                confirm = False
                while not confirm:
                    try:
                        packet_in = socket.recv_pyobj(zmq.NOBLOCK)
                        confirm = True
                    except zmq.error.Again:
                        pass
                if not packet_in.message == "receiver":
                    if VERBOSE:
                        print("SENDER: Something went wrong, exiting!")
                    running = False
                else:
                    if VERBOSE:
                        print("SENDER: Receiver confirmed")
                    data_in = packet_in.data
                    if data_in:
                        if "exit" in data_in.keys():
                            if VERBOSE:
                                print("Exit received!")
                            raise KeyboardInterrupt

                # CHECK: if maximum iterations reached...
                cur_processing += 1
                if max_processing != -1:
                    if cur_processing >= max_processing:
                        raise KeyboardInterrupt

        except KeyboardInterrupt:
            if VERBOSE:
                print("SENDER: Someone wants to killlllll meeeeeeeeeeee!")

            # CTRL-C has pressed, try to stop other process
            if VERBOSE:
                print("SENDER: Generating and sending packet...")
            packet_out = DataPacket(message="stop")
            socket.send_pyobj(packet_out)

            # CONFIRMATION: Loops and waits for confirmation that data was received
            if VERBOSE:
                print("SENDER: Waiting for reception confirmation...")
            confirm = False
            while not confirm:
                try:
                    packet_in = socket.recv_pyobj(zmq.NOBLOCK)
                    confirm = True
                except zmq.error.Again:
                    pass

            if not packet_in.message == "receiver":
                if VERBOSE:
                    print("SENDER: Something went wrong, but we're exiting anyway!")

    # exit gracefully...
    if VERBOSE:
        print("Closing socket and destroying context!")
    socket.close()
    context.destroy()
    if VERBOSE:
        print("Done with everything!")


if __name__ == "__main__":

    VERBOSE: bool = True
    url: str = "tcp://localhost:5555"  # number specifies communication channel

    class_processing = TestClassAnimate(show_horizontal_bar=True, wait_interval=5)
    sender(
        url=url, sender_processing=class_processing, max_processing=3, VERBOSE=VERBOSE
    )
    del class_processing


# %%
