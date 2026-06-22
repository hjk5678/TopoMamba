# utils/topology_configs.py


def get_topology_pairs(dataset):
    """
    返回每个数据集的 mstc 拓扑/连通性监督通道定义。

    每个元素格式：
        (name, class_a_set, class_b_set)

    说明：
        class_a_set is None and class_b_set is None:
            表示任意语义边界 any semantic boundary。

        class_a_set / class_b_set 为 list:
            表示这两个类别集合之间的边界。

    注意：
        GID 的类别编号可能因你的数据预处理而不同。
        如果你的 GID label map 不一样，需要手动改这里。
    """

    dataset = dataset.lower()

    # ------------------------------------------------------------
    # ISPRS Potsdam / Vaihingen
    # 常见类别顺序：
    # 0 impervious_surface
    # 1 building
    # 2 low_vegetation
    # 3 tree
    # 4 car
    # 5 clutter
    # ------------------------------------------------------------
    if dataset in ["potsdam", "vaihingen"]:
        return [
            ("any_boundary", None, None),
            ("low_vegetation_tree", [2], [3]),
            ("clutter_vegetation", [5], [2, 3]),
            ("clutter_impervious", [5], [0]),
        ]

    # ------------------------------------------------------------
    # LoveDA
    # 这里按你之前代码中的类别数 7 设计：
    # 0 background
    # 1 building
    # 2 road
    # 3 water
    # 4 barren
    # 5 forest
    # 6 agriculture
    # ------------------------------------------------------------
    elif dataset == "loveda":
        return [
            ("any_boundary", None, None),
            ("building_road", [1], [2]),
            ("forest_agriculture", [5], [6]),
            ("barren_agriculture", [4], [6]),
            ("water_nonwater", [3], [1, 2, 4, 5, 6]),
        ]

    # ------------------------------------------------------------
    # GID-5 假设类别顺序：
    # 0 built-up
    # 1 farmland
    # 2 forest
    # 3 meadow
    # 4 water
    #
    # 如果你的 GID 不是这个顺序，需要改这里。
    # ------------------------------------------------------------
    elif dataset in ["gid", "gid5"]:
        return [
            ("any_boundary", None, None),
            ("built_farmland", [0], [1]),
            ("forest_meadow", [2], [3]),
            ("water_nonwater", [4], [0, 1, 2, 3]),
        ]

    # ------------------------------------------------------------
    # 默认配置：最安全，只做任意语义边界
    # ------------------------------------------------------------
    else:
        return [
            ("any_boundary", None, None),
        ]


def describe_topology_pairs(topology_pairs):
    lines = []
    for i, item in enumerate(topology_pairs):
        name, class_a_set, class_b_set = item
        lines.append(
            f"channel {i}: {name}, class_a={class_a_set}, class_b={class_b_set}"
        )
    return "\n".join(lines)