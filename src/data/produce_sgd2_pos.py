import argparse
import s_gd2
import numpy as np
if __name__ == '__main__':
    parser  = argparse.ArgumentParser("produce_sgd2_pos",description="generate s_gd2 layout")
    parser.add_argument("--input",required=True,help="input topology")
    parser.add_argument("--output",required=True,help="output")
    args = parser.parse_args()
    G = np.load(args.input)
    pos = s_gd2.layout(G[0],G[1])
    np.save(args.output,pos)