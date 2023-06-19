class InfiniteDataLoader(object):
    def __init__(self, dataloader):
        super(InfiniteDataLoader, self).__init__()
        self.dataloader = dataloader
        self.data_iter = iter(self.dataloader)

    def next(self):
        try:
            data = next(self.data_iter)
        except StopIteration:
            # Reached end of the dataset
            self.data_iter = iter(self.dataloader)
            data = next(self.data_iter)

        return data

    def __len__(self):
        return len(self.dataloader)
