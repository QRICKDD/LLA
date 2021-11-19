from optparse import OptionParser
def get_option(MODE):
    if MODE == 'Librispeech':
        cfg = r"prefile/SincNet_LB.cfg"
        mp = r"prefile/model_lb_raw.pkl"
    elif MODE == 'TIMIT':
        cfg = r"prefile/timit_speaker.cfg"
        mp = r"prefile/model_raw.pkl"
    args = {'file': 'helloyouneedme', 'local_rank': 0, 'speaker_model': mp,
            'channel': [32, 32, 32, 32, 32], 'kernel_size': [3, 3, 3, 3, 3], 'dilation': [1, 2, 5, 2, 1],
            'sample': [1, 1, 1, 1, 1], 'speaker_cfg': 'prefile/SincNet_LB.cfg'}
    parser = OptionParser()
    parser.add_option("-f", "--file", default="helloyouneedme", help="helloyouneedme")
    parser.add_option("--local_rank", type=int, default=0)
    parser.add_option("--speaker_model", type=str, default=mp, help="path for pretrained speaker model")
    parser.add_option("--channel", type=int, nargs='+', default=[32, 32, 32, 32, 32],
                      help="channel for tranformer model")
    parser.add_option("--kernel_size", type=int, nargs='+', default=[3, 3, 3, 3, 3],
                      help="kernel size for transformer model")
    parser.add_option("--dilation", type=int, nargs='+', default=[1, 2, 5, 2, 1], help="dilation for transformer model")
    parser.add_option("--sample", type=int, nargs='+', default=[1, 1, 1, 1, 1], help="sample for transformer model")
    parser.add_option("--speaker_cfg", type=str, default=cfg, help="")
    args = parser.parse_args()[0]
    print(type(args))
    return args
