#######################################################
##                 merge ply files                   ##
#######################################################
# URL: https://qiita.com/tttamaki/items/648422860869bbccc72d
# ENV: python3.6, open3d-python==0.7.0

import sys
sys.path.append("../..") # Open3D/build/lib/ へのパス
import numpy as np
import open3d as py3d

RANSAC = py3d.registration_ransac_based_on_feature_matching
ICP = py3d.registration_icp
FPFH = py3d.compute_fpfh_feature
GET_GTG = py3d.get_information_matrix_from_point_clouds

def register(pcd1, pcd2, size):
    # ペアの点群を位置合わせ

    kdt_n = py3d.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    kdt_f = py3d.KDTreeSearchParamHybrid(radius=size * 10, max_nn=50)

    # ダウンサンプリング
    pcd1_d = py3d.voxel_down_sample(pcd1, size)
    pcd2_d = py3d.voxel_down_sample(pcd2, size)
    py3d.estimate_normals(pcd1_d, kdt_n)
    py3d.estimate_normals(pcd2_d, kdt_n)

    # 特徴量計算
    pcd1_f = FPFH(pcd1_d, kdt_f)
    pcd2_f = FPFH(pcd2_d, kdt_f)

    # 準備
    checker = [py3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
               py3d.CorrespondenceCheckerBasedOnDistance(size * 2)]

    est_ptp = py3d.TransformationEstimationPointToPoint()
    est_ptpln = py3d.TransformationEstimationPointToPlane()

    criteria = py3d.RANSACConvergenceCriteria(max_iteration=400000,
                                              max_validation=500)
    # RANSACマッチング
    result1 = RANSAC(pcd1_d, pcd2_d,
                     pcd1_f, pcd2_f,
                     max_correspondence_distance=size * 2,
                     estimation_method=est_ptp,
                     ransac_n=4,
                     checkers=checker,
                     criteria=criteria)
    # ICPで微修正
    result2 = ICP(pcd1, pcd2, size, result1.transformation, est_ptpln)

    return result2.transformation


def merge(pcds):
    # 複数の点群を1つの点群にマージする

    all_points = []
    for pcd in pcds:
        all_points.append(np.asarray(pcd.points))

    merged_pcd = py3d.PointCloud()
    merged_pcd.points = py3d.Vector3dVector(np.vstack(all_points))

    return merged_pcd


def add_color_normal(pcd): # in-place coloring and adding normal
    pcd.paint_uniform_color(np.random.rand(3))
    size = np.abs((pcd.get_max_bound() - pcd.get_min_bound())).max() / 30
    kdt_n = py3d.KDTreeSearchParamHybrid(radius=size, max_nn=50)
    py3d.estimate_normals(pcd, kdt_n)


def load_pcds(pcd_files):

    pcds = []
    for f in pcd_files:
        pcd = py3d.read_point_cloud(f)
        # add_color_normal(pcd)
        pcds.append(pcd)


    return pcds


def align_pcds(pcds, size):
    # 複数の点群を位置合わせ

    pose_graph = py3d.PoseGraph()
    accum_pose = np.identity(4) # id0から各ノードへの累積姿勢
    pose_graph.nodes.append(py3d.PoseGraphNode(accum_pose))

    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            source = pcds[source_id]
            target = pcds[target_id]

            trans = register(source, target, size)
            GTG_mat = GET_GTG(source, target, size, trans) # これが点の情報を含む

            if target_id == source_id + 1: # 次のidの点群ならaccum_poseにposeを積算
                accum_pose = trans @ accum_pose
                pose_graph.nodes.append(py3d.PoseGraphNode(np.linalg.inv(accum_pose))) # 各ノードは，このノードのidからid0への変換姿勢を持つので，invする
                # そうでないならnodeは作らない
            pose_graph.edges.append(py3d.PoseGraphEdge(source_id,
                                                       target_id,
                                                       trans,
                                                       GTG_mat,
                                                       uncertain=True)) # bunnyの場合，隣でも怪しいので全部True


    # 設定
    solver = py3d.GlobalOptimizationLevenbergMarquardt()
    criteria = py3d.GlobalOptimizationConvergenceCriteria()
    option = py3d.GlobalOptimizationOption(
             max_correspondence_distance=size / 10,
             edge_prune_threshold=size / 10,
             reference_node=0)

    # 最適化
    py3d.global_optimization(pose_graph,
                            method=solver,
                            criteria=criteria,
                            option=option)

    # 推定した姿勢で点群を変換
    for pcd_id in range(n_pcds):
        trans = pose_graph.nodes[pcd_id].pose
        pcds[pcd_id].transform(trans)


    return pcds



pcds = load_pcds(["../PLY/ukulele0.ply",
                  "../PLY/ukulele1.ply",
                  "../PLY/ukulele0.ply",
                  "../PLY/ukulele1.ply"])
py3d.draw_geometries(pcds, "input pcds")

size = np.abs((pcds[0].get_max_bound() - pcds[0].get_min_bound())).max() / 30

pcd_aligned = align_pcds(pcds, size)
py3d.draw_geometries(pcd_aligned, "aligned")

pcd_merge = merge(pcd_aligned)
# add_color_normal(pcd_merge)
py3d.io.write_point_cloud("combined_all_uke.pcd", pcd_merge)
py3d.draw_geometries([pcd_merge], "merged")