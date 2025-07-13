import numpy as np
INDEX = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,"8": 8,
         "9": 9, "10": 10, "11": 11, "12": 12, "13": 13,"14": 14, "15": 15, "16": 16,
         "17": 1, "18": 2, "19": 3, "20": 4, "21": 5, "22": 6,"23": 7, "24": 8,
         "25": 9, "26": 10, "27": 11, "28": 12, "29": 13,"30": 14, "31": 15, "32": 16}
INDEX_UP = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,"8": 8,
         "9": 9, "10": 10, "11": 11, "12": 12, "13": 13,"14": 14, "15": 15, "16": 16}
INDEX_DOWN = {"17": 1, "18": 2, "19": 3, "20": 4, "21": 5, "22": 6,"23": 7, "24": 8,
         "25": 9, "26": 10, "27": 11, "28": 12, "29": 13,"30": 14, "31": 15, "32": 16}

ROTAXIS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
decoder_r = np.zeros((1, 4), np.float32).repeat(14, axis=0)
decoder_t = np.zeros((1, 3), np.float32).repeat(14, axis=0)
Angles = [i for i in range(0, 31, 2)]
AgSize = len(Angles)

dim = 3
teeth_nums = 16
sam_points = 512

is_handle_up = True

gpu_divice_id = "0"

AUC_K = 5
AUC_piece = 0.05

absolute_path_prex = '/home/charon/codeGala/dzx/orth-tooth/'

VIEW_NUMS = 1

SAVE_MODEL = 50