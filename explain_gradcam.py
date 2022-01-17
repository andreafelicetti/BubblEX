import os
import argparse

import numpy
import torch
import torch.nn as nn
from data import S3DIS, ArCH, Sinthcity, ModelNet40, ShapeNetPart
from model import DGCNN_semseg, DGCNN_cls
from torch.utils.data import DataLoader
from gradcam_exp import gradcam

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

import pickle


def extract(args):
    total_areas = 0
    if args.dataset == "S3DIS":
        total_areas = 6
        args.num_classes = 13
    elif args.dataset == "ArCH":
        total_areas = 17
        args.num_classes = 10
    elif args.dataset == "ArCH9l":
        total_areas = 17
        args.num_classes = 9
    elif args.dataset == "synthcity":
        total_areas = 9
        args.num_classes = 9

    for test_area in range(1, total_areas + 1):
        test_area = str(test_area)
        if (args.test_area == 'all') or (test_area == args.test_area):
            if args.dataset == "S3DIS":
                test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "ArCH":
                test_loader = DataLoader(ArCH(partition='test', num_points=args.num_points, test_area=test_area),
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "ArCH9l":
                test_loader = DataLoader(
                    ArCH(partition='test', num_points=args.num_points, test_area=test_area, tipo="9l"),
                    batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "synthcity":
                test_loader = DataLoader(Sinthcity(partition='test', num_points=args.num_points, test_area=test_area),
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "modelnet40":
                test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            elif args.dataset == "shapenetpart":
                test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points),
                                         batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            else:
                print("Dataset not defined.")
                return

            if not args.no_cuda:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Try to load models
            if args.model == 'dgcnn':
                model = DGCNN_semseg(args).to(device)
            else:
                raise Exception("Not implemented")

            if args.parallel:
                model = nn.DataParallel(model)

            if args.model_path == "":
                print(os.path.join(args.model_root, 'model_%s.t7' % test_area))
                model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % test_area)))
            else:
                model.load_state_dict(torch.load(os.path.join(args.model_path)))

            model = model.train()

            cam = gradcam.GradCAM(model=model,
                          target_layer=model.conv9,
                          use_cuda=True)

            print("Model defined...")
            i = 0
            for data, seg, max in test_loader:
                data = data.permute(0, 2, 1).to(device)
                res = cam(input_tensor=data,
                          target_category=6,
                          aug_smooth=False,
                          eigen_smooth=False)
                res_one = res[:, :].squeeze()
                torch.set_printoptions(edgeitems=20, sci_mode=False)
                numpy.set_printoptions(edgeitems=20, suppress=True)

                x = data[0, 0]
                y = data[0, 2]
                z = data[0, 1]
                #res_one = res_one.reshape((1,4096))
                isNaN = numpy.isnan(res_one).any()
                if not isNaN:
                    #print(res_one)
                    ply = numpy.stack((x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()), axis=-1)
                    pcd = o3d.geometry.PointCloud()
                    # res_one = res_one.reshape((4096, 1))
                    res_one = numpy.stack((res_one, numpy.zeros_like(res_one), numpy.zeros_like(res_one)), axis=-1)
                    pcd.points = o3d.utility.Vector3dVector(ply)
                    pcd.colors = o3d.utility.Vector3dVector(res_one)
                    o3d.io.write_point_cloud("./res/data" + str(i) + ".ply", pcd)
                i += 1


def extract_cls(args):
    # total_areas = 0
    # if args.dataset == "S3DIS":
    #     total_areas = 6
    #     args.num_classes = 13
    # elif args.dataset == "ArCH":
    #     total_areas = 17
    #     args.num_classes = 10
    # elif args.dataset == "ArCH9l":
    #     total_areas = 17
    #     args.num_classes = 9
    # elif args.dataset == "synthcity":
    #     total_areas = 9
    #     args.num_classes = 9
    # elif args.dataset == "modelnet40":
    #     total_areas = 1
    #     args.num_classes = 40
    #
    # print(args.dataset)
    # for test_area in range(1, total_areas + 1):
    #     test_area = str(test_area)
    #     if (args.test_area == 'all') or (test_area == args.test_area):
    #         if args.dataset == "S3DIS":
    #             test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
    #                                      batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    #         elif args.dataset == "ArCH":
    #             test_loader = DataLoader(ArCH(partition='test', num_points=args.num_points, test_area=test_area),
    #                                      batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    #         elif args.dataset == "ArCH9l":
    #             test_loader = DataLoader(
    #                 ArCH(partition='test', num_points=args.num_points, test_area=test_area, tipo="9l"),
    #                 batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    #         elif args.dataset == "synthcity":
    #             test_loader = DataLoader(Sinthcity(partition='test', num_points=args.num_points, test_area=test_area),
    #                                      batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    #         elif args.dataset == "modelnet40":
    #             test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
    #                                      batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    #         elif args.dataset == "shapenetpart":
    #             test_loader = DataLoader(ShapeNetPart(partition='test', num_points=args.num_points),
    #                                      batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    #         else:
    #             print("Dataset not defined.")
    #             return

            ###
            if True:
                if True:
                    objs = np.load("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\objs.npy")
                    labs = np.genfromtxt("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\gt.txt", delimiter=' ').astype("int64")

                    #objs[:,:,[1,2]] = objs[:,:,[2,1]]
                    test_loader= zip(objs,labs) ##
                    #####

            if not args.no_cuda:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Try to load models
            if args.model == 'dgcnn_cls':
                model = DGCNN_cls(args).to(device)
            else:
                raise Exception("Not implemented")

            model = nn.DataParallel(model)

            print(os.path.join(args.model_path))
            if args.model_path == "":
                print(os.path.join(args.model_root, 'model_%s.t7' % test_area))
                model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % test_area)))
            else:
                if not args.no_cuda:
                    model.load_state_dict(torch.load(os.path.join(args.model_path)))
                else:
                    model.load_state_dict(torch.load(os.path.join(args.model_path),map_location = torch.device('cpu')))

            model = model.train()

            if not args.no_cuda:
                cam = gradcam.GradCAM(model=model,
                                      target_layer=model.module.conv5,#linear2,#conv5,
                                      use_cuda=True)
            else:
                cam = gradcam.GradCAM(model=model,
                                      target_layer=model.module.conv5,#linear2,#conv5,
                                      use_cuda=False)

            print("Model defined...")
            #i = 0

            # results = []
            # maxes = []
            # mines = []
            # info= []
            # RESULTS = []
            # MAXES = []
            # MINES = []
            # INFO = []

            for cls in range(0,args.num_classes)[:2]:
                i=0

                # results = []
                # maxes = []
                # mines = []
                # info = []

                GT = []
                # PRED = []
                # ID_DATA = []
                # DATA= []
                # ACT_P= []
                # ACT_N = []
                # GRAD_P= []
                # GRAD_N = []
                # #CAM = []
                # GRAD_MEDIAN= []
                # GRAD_MEANSQ= []
                AG_P= []
                AG_N= []
                AG_MAX= []
                AG_MIN= []
                AG_MED= []
                AG_MS= []

                for data, gt in test_loader:
                    data = data.permute(0, 2, 1).to(device)
                    # res, min_v, max_v, output, act_p, act_n,g_p, g_n = cam(input_tensor=data,
                    #           target_category=cls,
                    #           aug_smooth=False,
                    #           eigen_smooth=False)
                    # res, min_v, max_v, output, g_median, g_meansq = cam(input_tensor=data,
                    #                                                         target_category=cls,
                    #                                                         aug_smooth=False,
                    #                                                         eigen_smooth=False)
                    output, ag_p, ag_n, ag_max, ag_min, ag_med, ag_ms = cam(input_tensor=data,
                                                                            target_category=cls,
                                                                            aug_smooth=False,
                                                                            eigen_smooth=False)
                    # res_one = res[:, :].squeeze()
                    torch.set_printoptions(edgeitems=20, sci_mode=False)
                    numpy.set_printoptions(edgeitems=20, suppress=True)

                    out = torch.argmax(output.squeeze())

                    #results.append((data, res_one, max, out ))
                    # results.append((data, res_one, max, act_p, act_n,g_p, g_n ))
                    # maxes.append(max_v)
                    # mines.append(min_v)
                    # info.append((cls,i))

                    GT.append(gt.numpy()[0, 0])
                    # PRED.append(int(out))
                    # ID_DATA.append(i)
                    # DATA.append(data.numpy()[0, :].T)
                    # ACT_P.append(act_p[0, :])
                    # ACT_N.append(act_n[0, :])
                    # GRAD_P.append(g_p[0, :])
                    # GRAD_N.append(g_n[0, :])
                    # CAM.append(res_one)
                    # GRAD_MEDIAN.append(g_median[0, :])
                    # GRAD_MEANSQ.append(g_meansq[0, :])
                    AG_P.append(ag_p[0, :])
                    AG_N.append(ag_n[0, :])
                    AG_MAX.append(ag_max[0, :])
                    AG_MIN.append(ag_min[0, :])
                    AG_MED.append(ag_med[0, :])
                    AG_MS.append(ag_ms[0, :])

                    #print("mean_tg{}_id{} = {}".format(cls,i,numpy.mean(res)))
                    print(i)

                    i+=1

                    if i==2:
                        break


                GT = np.array(GT)
                # PRED = np.array(PRED)
                # ID_DATA = np.array(ID_DATA)
                # DATA = np.array(DATA)
                # ACT_P = np.array(ACT_P)
                # ACT_N = np.array(ACT_N)
                # GRAD_P = np.array(GRAD_P)
                # GRAD_N = np.array(GRAD_N)
                #CAM = np.array(CAM)
                # GRAD_MEDIAN= np.array(GRAD_MEDIAN)
                # GRAD_MEANSQ= np.array(GRAD_MEANSQ)
                AG_P = np.array(AG_P)
                AG_N = np.array(AG_N)
                AG_MAX = np.array(AG_MAX)
                AG_MIN = np.array(AG_MIN)
                AG_MED = np.array(AG_MED)
                AG_MS = np.array(AG_MS)


                print("class " + str(cls) + " DONE")

                # save CHECKPOINT
                vDict = {}
                # vDict["results"] = results
                # vDict["maxes"] = maxes
                # vDict["mines"] = mines
                # vDict["info"] = info

                vDict["GT"] = GT
                # vDict["PRED"] = PRED
                # vDict["ID_DATA"] = ID_DATA
                # vDict["DATA"] = DATA
                # vDict["ACT_P"] = ACT_P
                # vDict["ACT_N"] = ACT_N
                # vDict["GRAD_P"] = GRAD_P
                # vDict["GRAD_N"] = GRAD_N
                #vDict["CAM"] = CAM
                # vDict["GRAD_MEDIAN"] = GRAD_MEDIAN
                # vDict["GRAD_MEANSQ"] = GRAD_MEANSQ
                vDict["AG_P"] = AG_P
                vDict["AG_N"] = AG_N
                vDict["AG_MAX"] = AG_MAX
                vDict["AG_MIN"] = AG_MIN
                vDict["AG_MED"] = AG_MED
                vDict["AG_MS"] = AG_MS

                with open("gramcamCK_tg{}.pkl".format(cls), "wb") as f:
                    pickle.dump(vDict, f)

                # RESULTS+= results
                # MAXES+= maxes
                # MINES+= mines
                # INFO+= info

                # if cls==3:
                #     break

# # save CHECKPOINT
# vDict={}
# vDict["results"] = results
# vDict["maxes"] = maxes
# vDict["mines"] = mines
# vDict["info"] = info
# with open("gramcamCK.pkl", "wb") as f:
#     pickle.dump(vDict, f)

# # load CHECKPOINT
# with open("gramcamCK.pkl", "rb") as f:
#     vDict = pickle.load(f)
# results = vDict["results"]
# maxes= vDict["maxes"]
# mines= vDict["mines"]
# info = vDict["info"]

                # # load CHECKPOINT
                # with open("gramcamCK_tg{}.pkl".format(cls), "rb") as f:
                #     vDict = pickle.load(f)
                # results = vDict["results"]
                # maxes= vDict["maxes"]
                # mines= vDict["mines"]
                # info = vDict["info"]

#max_v = numpy.max(maxes)
#min_v = numpy.min(mines)


            # results=RESULTS
            # maxes= MAXES
            # mines= MINES
            # info= INFO
            #
            # directory = "./res/"
            # if not os.path.exists(directory):
            #     os.makedirs(directory)
            #
            # i=0
            # for data, res, max, out, _,_ in results:
            #
            #     min_v = mines[i]
            #     max_v = maxes[i]
            #     cls,idx = info[i]
            #
            #     res_one = (res-min_v)/(max_v-min_v+0.000001)
            #
            #     x = data[0, 0]
            #     y = data[0, 2]
            #     z = data[0, 1]
            #     # res_one = res_one.reshape((1,4096))
            #     # print(res_one)
            #     ply = numpy.stack((x.cpu().numpy(), y.cpu().numpy(), z.cpu().numpy()), axis=-1)
            #     pcd = o3d.geometry.PointCloud()
            #     # res_one = res_one.reshape((4096, 1))
            #     #res_one = numpy.stack((res_one, numpy.zeros_like(res_one), numpy.zeros_like(res_one)), axis=-1)
            #
            #     cmap = plt.cm.get_cmap("jet")
            #     res_one = cmap(res_one)[:,:3]
            #
            #     pcd.points = o3d.utility.Vector3dVector(ply)
            #     pcd.colors = o3d.utility.Vector3dVector(res_one)
            #
            #     o3d.io.write_point_cloud(directory + "data_tg" + str(cls) + "_id" + str(idx) + "_gt" + str(max.item()) + "_p" + str(out.item())  + ".ply", pcd)
            #
            #     i += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Part Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'dgcnn_cls'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='synthcity', metavar='N',
                        choices=['S3DIS', "ArCH", "ArCH9l", "synthcity", "modelnet40"])
    parser.add_argument('--test_area', type=str, default='3', metavar='N')
    # choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cos', metavar='N',
                        choices=['cos', 'step'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', type=bool, default=True, #False
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--extract', type=bool, default=False,
                        help='feature extraction')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_root', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--parallel', type=bool, default=False,
                        help='Use Multiple GPUs')

    args = parser.parse_args()
    print("Extracting...")
    if args.model == 'dgcnn_cls':
        extract_cls(args)
    else:
        extract(args)
    print("End extraction.")
