"""Plot completed training epochs; no test metrics or projected results."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR',str((args.run/'mplcache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[json.loads(line) for line in (args.run/'epochs.jsonl').read_text('utf-8').splitlines() if line.strip()]
    x=[r['epoch']+1 for r in records]
    fig,axes=plt.subplots(1,3,figsize=(14,3.7),layout='constrained')
    axes[0].plot(x,[r['train_loss'] for r in records],marker='o')
    axes[0].set_title('Training loss (two heads)')
    for name,label in (('species_top1','Top-1'),('species_macro_f1','Macro-F1')):
        axes[1].plot(x,[r['validation'][name] for r in records],marker='o',label=label)
    axes[1].set_title('Grouped validation species'); axes[1].set_ylim(0,1); axes[1].legend()
    for name,label in (('tick_recall','Tick recall'),('non_tick_fpr','Non-tick proxy FPR')):
        axes[2].plot(x,[r['validation'][name] for r in records],marker='o',label=label)
    axes[2].set_title('Grouped validation presence'); axes[2].set_ylim(0,1); axes[2].legend()
    for ax in axes:
        ax.set_xlabel('Completed epoch'); ax.grid(alpha=.25)
    fig.suptitle('Component validation only; correlated views; default system unchanged')
    fig.savefig(args.run/'learning_curves.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    main()
