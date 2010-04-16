from recon.operations import Operation, ChannelAwareOperation

class CombineChannels (Operation):
    @ChannelAwareOperation
    def run(self, image):
        image.combine_channels()
