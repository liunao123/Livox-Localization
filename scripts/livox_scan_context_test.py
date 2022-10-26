#!/usr/bin/env python2

import numpy as np

from scan_context_manager import ScanContextManager
import os

if __name__ == "__main__":
    # date_path = "/home/map/1012_2205/SC/"
    # date_path = "/home/map/1012_2205/"
    date_path = "/home/map/0928_1220/"

    Livox_SC = ScanContextManager(file_path=date_path + "SC")

    pcd_path = os.path.join(date_path, "odo_save")
    sc_path = os.path.join(date_path, "SC/scancontext")
    rk_path = os.path.join(date_path, "SC/ringkey")
    npy_path = os.path.join(date_path, "SC/pcd_npy")
    
    folder = os.path.exists(pcd_path) 
    if not folder:
        os.makedirs(pcd_path) 
    
    folder = os.path.exists(sc_path) 
    if not folder:
        os.makedirs(sc_path) 

    folder = os.path.exists(rk_path) 
    if not folder:
        os.makedirs(rk_path) 
    
    folder = os.path.exists(npy_path) 
    if not folder:
        os.makedirs(npy_path)

    make_sc = True
    # make_sc = False
    if make_sc:
        # Test the ScanContext Maker
        # Livox_SC.livox_load_pc_make_sc(pcd_path)
        Livox_SC.livox_load_pc_make_sc_from_pose_graph( date_path )

    else:
        # Test the ScanContext Load and Localization
        Livox_SC.livox_load_sc_rk(sc_path, rk_path)
        for i in range(0, 500, 50):
            file_name = "pc_" + str(i) + ".npy"
            print("load test pc: ", file_name)
            test_pc = os.path.join(npy_path, file_name)
            test_trans = Livox_SC.initialization(np.load(test_pc))
            print(test_trans)
