from typing import List
# import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from LogReg import LogReg, RegularizationParams

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x, **kwargs: x


def task1():
    # Define params
    logreg_params = [
            {
                "data_idx": i,
                "regularization": None,  #  RegularizationParams(norm=2, lambda_=1),
                "feature_map": None
            } for i in range(1, 5)
        ]
    result_dict = {}

    for idx, params in tqdm(enumerate(logreg_params)):
        # Load and train
        lreg = LogReg(*params.values())
        # lreg.train()
        with np.errstate(invalid='ignore', divide="ignore"):
            lreg.train()

        # Get results
        result_dict["Dataset "+str(params['data_idx'])] = {
                "train_acc": lreg.trainAcc(),
                "val_acc": lreg.valAcc(),
                "test_acc": lreg.testAcc(),
                }

        # Plot in subfigure
        plt.subplot(2, 2, idx+1)
        lreg.plotDecisionBoundary(title='Dataset ' f'{idx} with decision boudary and\n' +
                                        ('no regularization' if params['regularization'] is None
                                        else (f'$L_{params["regularization"].norm}$ norm and '
                                            '$\lambda=' f'{params["regularization"].lambda_:.1G}$'))
                                        ) 

    # Print and show results
    result_df = pd.DataFrame.from_dict(result_dict, orient='columns')
    print(result_df)

    plt.tight_layout()
    plt.show()

def main():
    task1()

if __name__ == "__main__":
    main()
