import monai
monai.utils.set_determinism(2000)

def get_model():
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(8, 16, 32, 64, 128),
        strides=(2, 2, 2, 2),
        num_res_units=3,
        dropout=0.3
    )
    return model