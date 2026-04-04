import argparse
import os
import traceback
import time
from multiprocessing import Process
import multiprocessing

import pandas as pd

from warnings import filterwarnings
filterwarnings("ignore")

from model.model import PatchBPAD
from utils.dataset import Dataset
from utils.eval import cal_best_PRF
from utils.fs import EVENTLOG_DIR, ROOT_DIR


def fit_and_eva(dataset_name, ad, fit_kwargs=None, ad_kwargs=None):
    if ad_kwargs is None:
        ad_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    start_time = time.time()

    print(dataset_name)
    dataset = Dataset(dataset_name)

    ad = ad(**ad_kwargs)
    print(ad.name)
    resPath = os.path.join(ROOT_DIR, f'results/result_{ad.name}.csv')
    try:
        ad.fit(dataset, **fit_kwargs)

        trace_level_abnormal_scores, event_level_abnormal_scores, attr_level_abnormal_scores = ad.detect(dataset)

        end_time = time.time()
        run_time = end_time - start_time
        print('run_time')
        print(run_time)

        ## trace level
        trace_p, trace_r, trace_f1, trace_aupr = cal_best_PRF(dataset.case_target, trace_level_abnormal_scores)
        print("Trace-level anomaly detection")
        print(f'precision: {trace_p}, recall: {trace_r}, F1-score: {trace_f1}, AP: {trace_aupr}')

        if event_level_abnormal_scores is not None:
            ## event level
            eventTemp = dataset.binary_targets.sum(2).flatten()
            eventTemp[eventTemp > 1] = 1
            event_p, event_r, event_f1, event_aupr = cal_best_PRF(eventTemp, event_level_abnormal_scores.flatten())
            print("Event-level anomaly detection")
            print(f'precision: {event_p}, recall: {event_r}, F1-score: {event_f1}, AP: {event_aupr}')
        else:
            event_p, event_r, event_f1, event_aupr = 0, 0, 0, 0

        ## attr level
        if attr_level_abnormal_scores is not None:
            attr_p, attr_r, attr_f1, attr_aupr = cal_best_PRF(dataset.binary_targets.flatten(),
                                                                attr_level_abnormal_scores.flatten())
            print("Attribute-level anomaly detection")
            print(f'precision: {attr_p}, recall: {attr_r}, F1-score: {attr_f1}, AP: {attr_aupr}')
        else:
            attr_p, attr_r, attr_f1, attr_aupr = 0, 0, 0, 0

        datanew = pd.DataFrame([{'index': dataset_name,
                                  'trace_p': trace_p, "trace_r": trace_r, 'trace_f1': trace_f1, 'trace_aupr': trace_aupr,
                                  'event_p': event_p, "event_r": event_r, 'event_f1': event_f1, 'event_aupr': event_aupr,
                                  'attr_p': attr_p, "attr_r": attr_r, 'attr_f1': attr_f1, 'attr_aupr': attr_aupr, 'time': run_time
                                  }])
        if os.path.exists(resPath):
            data = pd.read_csv(resPath)
            data = pd.concat([data, datanew], ignore_index=True)
        else:
            data = datanew
        data.to_csv(resPath, index=False)
    except Exception as e:
        traceback.print_exc()
        datanew = pd.DataFrame([{'index': dataset_name}])
        if os.path.exists(resPath):
            data = pd.read_csv(resPath)
            data = pd.concat([data, datanew], ignore_index=True)
        else:
            data = datanew
        data.to_csv(resPath, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='PatchBPAD: Patch-based Business Process Anomaly Detection')

    # Dataset
    parser.add_argument('-d', '--datasets', nargs='+', default=None,
                        help='Dataset names to evaluate (default: all in eventlogs/)')

    # Model hyperparameters
    parser.add_argument('--window-size', type=int, default=3)
    parser.add_argument('--d-emb', type=int, default=16)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num-enc-layers', type=int, default=2)
    parser.add_argument('--num-dec-layers', type=int, default=2)
    parser.add_argument('--d-ff', type=int, default=128)
    parser.add_argument('--d-gru', type=int, default=64)
    parser.add_argument('--enc-dropout', type=float, default=0.3)
    parser.add_argument('--dec-dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--n-epochs', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    args = parse_args()

    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = os.listdir(EVENTLOG_DIR)
        dataset_names.sort()
        if 'cache' in dataset_names:
            dataset_names.remove('cache')

    ad_kwargs = dict(
        window_size=args.window_size,
        d_emb=args.d_emb,
        d_model=args.d_model,
        nhead=args.nhead,
        num_enc_layers=args.num_enc_layers,
        num_dec_layers=args.num_dec_layers,
        d_ff=args.d_ff,
        d_gru=args.d_gru,
        enc_dropout=args.enc_dropout,
        dec_dropout=args.dec_dropout,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    print('number of datasets:' + str(len(dataset_names)))
    for d in dataset_names:
        p = Process(target=fit_and_eva, kwargs={'dataset_name': d, 'ad': PatchBPAD, 'ad_kwargs': ad_kwargs})
        p.start()
        p.join()
