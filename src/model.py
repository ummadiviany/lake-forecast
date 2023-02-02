import monai

def get_model():
    model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2),
        num_res_units=2,
    )
    return model