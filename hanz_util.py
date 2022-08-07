class LossTracker:
    lossMean = None
    lossStd = None
    lossSamples = []
    lossSamplesToCollect = 20
    lossVelocity = 0
    start_timestamp = None
    def push(self, loss):
        self.lossSamples.append(loss.item())
        if(self.lossMean == None):
            self.lossMean = self.lossSamples[0]
            self.start_timestamp = time.time()
        if len(self.lossSamples) >= self.lossSamplesToCollect:
            lossMean2 = np.mean(self.lossSamples)
            lossStd2 = np.std(self.lossSamples)
            self.lossVelocity = lossMean2 - self.lossMean
            self.lossMean = lossMean2
            self.lossStd = lossStd2
            self.lossSamples = []
            self.display()
            self.start_timestamp = time.time()
    def display(self):
        t = time.time() - self.start_timestamp
        rate_changed = self.lossVelocity / self.lossMean
        target_rate_change = 0.1
        ratio = abs(target_rate_change / rate_changed)
        seconds = int(ratio * t)
        steps = int(ratio * self.lossSamplesToCollect)
        print("-- loss velocity in {} steps: {}".format(self.lossSamplesToCollect, self.lossVelocity))
        print("----- {} {}% in {} seconds ({} steps)".format('Reduce' if self.lossVelocity<0 else 'Increase',  target_rate_change*100, seconds, steps))
