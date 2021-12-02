import argparse
import torch_fidelity


def measure_visual_fidelity(args):
    metrics = torch_fidelity.calculate_metrics(
        input1=args.sim_path,
        input2=args.real_path,
        isc=True,
        fid=True,
        kid=True,
        verbose=False,
    )
    for k, v in metrics.items():
        print(f'{k:<40s}: {v}')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-path', type=str, required=True)
    parser.add_argument('--sim-path', type=str, required=True)

    args = parser.parse_args()

    measure_visual_fidelity(args)