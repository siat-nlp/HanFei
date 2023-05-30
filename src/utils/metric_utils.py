import torch
import torchmetrics


class AccumulatedLoss(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("loss", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor):
        # update metric states
        self.loss += loss
        self.total += 1

    def compute(self):
        # compute final result
        return self.loss.float() / self.total


if __name__ == '__main__':
    # class LitModel(LightningModule):
    #     def __init__(self):
    #         # 1. initialize the metric
    #         self.accumulated_loss = AccumulatedLoss()
    #
    #     def training_step(self, batch, batch_idx):
    #         x, y = batch
    #         loss = self(x)
    #
    #         # 2. compute the metric
    #         self.accumulated_loss(loss)
    #
    #         # 3. log it
    #         self.log("train_loss", self.accumulated_loss)
    #
    #     def training_epoch_end(self, outputs):
    #         # 4. reset the metric
    #         self.accumulated_loss.reset()
    pass
