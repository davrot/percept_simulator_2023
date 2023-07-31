#%%

from dataclasses import dataclass, field


@dataclass
class DataPacket:

    # data packet consists of message/command and data associated with it
    data: dict = field(default_factory=dict)
    message: str = field(default=None)


if __name__ == "__main__":
    test = DataPacket()
    print(test.data)
    print(test.message)


# %%
