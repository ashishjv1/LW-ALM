import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
import torch 

class binning():
    def __init__(self, preds, labels_oneh, num_bins=10, savename="calibrated_network.png", figsize=(5,5), title="Hists"):
        self.preds = preds
        self.labels_oneh = labels_oneh
        self.num_bins = num_bins
        self.save_name = savename
        self.figsize = figsize
        self.title = title
        self.draw_reliability_graph()
        
        
    def calc_bins(self):
        bins = np.linspace(0.1, 1, self.num_bins)
        binned = np.digitize(self.preds, bins)
        # Save the accuracy, confidence and size of each bin
        bin_accs = np.zeros(self.num_bins)
        bin_confs = np.zeros(self.num_bins)
        bin_sizes = np.zeros(self.num_bins)
        
        for bin in range(self.num_bins):
            bin_sizes[bin] = len(self.preds[binned == bin])
            if bin_sizes[bin] > 0:
                bin_accs[bin] = (self.labels_oneh[binned==bin]).sum() / bin_sizes[bin]
                bin_confs[bin] = (self.preds[binned==bin]).sum() / bin_sizes[bin]
        return bins, binned, bin_accs, bin_confs, bin_sizes


    def get_metrics(self):
        ECE = 0
        MCE = 0
        bins, _, bin_accs, bin_confs, bin_sizes = self.calc_bins()

        for i in range(len(bins)):
            abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
            ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
            MCE = max(MCE, abs_conf_dif)

        return ECE, MCE



    def draw_reliability_graph(self):
        ECE, MCE = self.get_metrics()
        bins, _, bin_accs, _, _ = self.calc_bins()

#         fig = plt.figure(figsize=(5, 5))
        fig = plt.figure(figsize=self.figsize)
        ax = fig.gca()

        # x/y limits
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1)

        # x/y labels
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')

        # Create grid
        ax.set_axisbelow(True) 
        ax.grid(color='gray', linestyle='dashed')

        # Error bars
        plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

        # Draw bars and identity line
        plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
        plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

        # Equally spaced axes
        plt.gca().set_aspect('equal', adjustable='box')

        # ECE and MCE legend
        ECE_patch = mpatches.Patch(color='None', label='ECE = {:.2f}%'.format(ECE*100))
        Outputs = mpatches.Patch(color='b', label='Outputs')
#         MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
        Gap = mpatches.Patch(color='r', label='Gap')
#         plt.legend(handles=[ECE_patch, MCE_patch])
        plt.legend(handles=[Outputs, Gap, ECE_patch])
#         plt.label("")

        #plt.show()
        plt.title(self.title)
        plt.savefig(self.save_name, bbox_inches='tight')

        #draw_reliability_graph(preds)
