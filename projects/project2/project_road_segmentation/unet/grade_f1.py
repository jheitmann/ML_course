
import argparse
from postprocessing import compute_trainset_f1
import utest

TEST_CSV_PATH = "results/output.csv"

if __name__=="__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("ckpt_path", type=str,
    #                    help="path to ckpt file")
    #args = parser.parse_args()
    #utest.main(args.ckpt_path, t=True)
    f1 = compute_trainset_f1(TEST_CSV_PATH)
    print("f1 score :", f1)

