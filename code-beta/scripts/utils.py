import os
import numpy as np


def load_bao_data(project_root=None):
    """
    Load BAO datasets including:
    - DV/rd measurements
    - DM/rd measurements
    - Reconstructed H(z)

    Parameters
    ----------
    project_root : str or None
        The absolute path to the project root.  
        If None, project_root is automatically set to the parent directory 
        of the current working directory (useful when running inside notebooks).

    Returns
    -------
    dict
        {
            "DV": {
                "z": array,
                "value": array,
                "error": array
            },
            "DM": {
                "z": array,
                "value": array,
                "error": array
            },
            "Hz": {
                "z": array (possibly None if not included),
                "value": array,
                "error": array
            }
        }
    """

    # 自动定位项目根目录（强烈推荐）
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output")

    # 构建正确路径
    file_DV = os.path.join(data_dir, "Combine_DV_rd.dat")
    file_DM = os.path.join(data_dir, "Combine_DM_rd.dat")
    file_Hz = os.path.join(output_dir, "Hz_DVreconstruct_results.txt")

    # ---- 数据读取 ----
    try:
        z_DV, DV_rd, DV_rd_err = np.loadtxt(file_DV, unpack=True)
        z_DM, DM_rd, DM_rd_err = np.loadtxt(file_DM, unpack=True)
        _, H_DV, H_DV_err = np.loadtxt(file_Hz, unpack=True)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"[ERROR] BAO data file not found: {e.filename}")

    # ---- 返回结构化字典 ----
    return {
        "DV": {"z": z_DV, "value": DV_rd, "error": DV_rd_err},
        "DM": {"z": z_DM, "value": DM_rd, "error": DM_rd_err},
        "Hz": {"z": None, "value": H_DV, "error": H_DV_err},
    }
